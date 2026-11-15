//! Complete bounded Jordan decompositions of literal rational matrices.
//!
//! Rational eigenvalues and their algebraic multiplicities are untrusted search
//! hints. For N = A - lambda I, size-k chain tops are chosen in
//! ker(N^k) / (ker(N^(k-1)) + N ker(N^(k+1))). Keeping this quotient is essential:
//! concatenating all previous chains with the smaller kernel introduces
//! dependent vectors and loses repeated or mixed nontrivial Jordan blocks.
//!
//! An independent reference checker never computes a kernel, rank, inverse or
//! spectrum. It checks canonical Jordan block structure, A P = P J and Q P = I
//! directly against every original rational cell. These identities establish
//! a complete similarity decomposition even if the search hints were wrong.
//! Merely checking some eigenvectors or chains would not establish completeness.
//!
//! Dimensions are at most sixteen, source/hint coefficients at most 4096 bits,
//! and intermediate/output coefficients at most 8192 bits. Separate generation
//! and verification phases each keep cumulative 200M-step / 512MiB allocation
//! envelopes through publication. This is an in-process checked capability,
//! not a new portable proof-kernel claim. Nonrational spectra are not admitted.

#![forbid(unsafe_code)]

use crate::{Matrix, MatrixError, numeric_value};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use num_traits::{One, Zero};
use std::cmp::Ordering;

pub const MAX_RATIONAL_JORDAN_DIMENSION: usize = 16;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 8192;
const MAX_STEPS: u64 = 200_000_000;
const MAX_BYTES: u64 = 512 * 1024 * 1024;
type Result<T> = std::result::Result<T, MatrixError>;

/// A standard Jordan block with ones on its superdiagonal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalJordanBlock {
    pub eigenvalue: BigRational,
    pub size: usize,
}

/// Untrusted complete similarity evidence. The inverse is a left-inverse witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalJordanCandidate {
    pub blocks: Vec<RationalJordanBlock>,
    pub transform: Matrix,
    pub inverse: Matrix,
}

/// Immutable source-bound evidence for A = P J P^-1 over the rationals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRationalJordanDecomposition {
    matrix: Matrix,
    certificate: RationalJordanCandidate,
    jordan: Matrix,
}

impl VerifiedRationalJordanDecomposition {
    pub fn matrix(&self) -> &Matrix {
        &self.matrix
    }
    pub fn transform(&self) -> &Matrix {
        &self.certificate.transform
    }
    pub fn inverse(&self) -> &Matrix {
        &self.certificate.inverse
    }
    pub fn jordan(&self) -> &Matrix {
        &self.jordan
    }
    pub fn certificate(&self) -> &RationalJordanCandidate {
        &self.certificate
    }
    /// Move the accepted (transform, Jordan matrix, inverse) without copying.
    pub fn into_parts(self) -> (Matrix, Matrix, Matrix) {
        (
            self.certificate.transform,
            self.jordan,
            self.certificate.inverse,
        )
    }
}

fn limit(reason: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("rational Jordan decomposition: {reason}"))
}
fn invalid(reason: &str) -> MatrixError {
    MatrixError::InvalidCertificate(format!("rational Jordan decomposition: {reason}"))
}
fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}
fn arithmetic(error: RationalArithmeticError) -> MatrixError {
    match error {
        RationalArithmeticError::Meter(error) => stopped(error),
        RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
        RationalArithmeticError::InvariantViolation(reason) => {
            MatrixError::ArithmeticInvariantViolation(reason)
        }
        error => MatrixError::ResourceLimit(error.to_string()),
    }
}
fn number_ok(value: &BigRational, maximum: u64) -> Result<()> {
    if value.numer().bits().max(value.denom().bits()) > maximum {
        return Err(limit("rational coefficient height"));
    }
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(invalid("invalid rational denominator"));
    }
    Ok(())
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
            stopped: None,
        }
    }
    fn storage(&mut self, steps: usize, bytes: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(stopped)
    }
    fn tick(&mut self, count: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, count.max(1) as u64)
            .map_err(stopped)
    }
    fn vector<T>(&mut self, count: usize) -> Result<Vec<T>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| limit("container size overflow"))?;
        self.storage(1, bytes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| limit("container allocation refused"))?;
        Ok(values)
    }
    fn zeros(&mut self, count: usize) -> Result<Vec<BigRational>> {
        let mut values = self.vector(count)?;
        self.storage(count, count.saturating_mul(64))?;
        values.resize_with(count, BigRational::zero);
        Ok(values)
    }
    fn zero(&mut self) -> Result<BigRational> {
        self.storage(1, 64)?;
        Ok(BigRational::zero())
    }
    fn one(&mut self) -> Result<BigRational> {
        self.storage(1, 64)?;
        Ok(BigRational::one())
    }
    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        number_ok(value, MAX_BITS)?;
        self.storage(
            1,
            64 + 8 * (value.numer().limb_count() + value.denom().limb_count()) as usize,
        )?;
        Ok(value.clone())
    }
    fn copies(&mut self, values: &[BigRational]) -> Result<Vec<BigRational>> {
        let mut result = self.vector(values.len())?;
        for value in values {
            result.push(self.copy(value)?);
        }
        Ok(result)
    }
    fn compare(&mut self, left: &BigRational, right: &BigRational) -> Result<Ordering> {
        number_ok(left, MAX_BITS)?;
        number_ok(right, MAX_BITS)?;
        self.checkpoint().map_err(stopped)?;
        left.metered_cmp(right, self).map_err(arithmetic)
    }
    fn combine(
        &mut self,
        left: &BigRational,
        right: &BigRational,
        operation: Operation,
    ) -> Result<BigRational> {
        number_ok(left, MAX_BITS)?;
        number_ok(right, MAX_BITS)?;
        let (an, ad, bn, bd) = (
            left.numer().bits(),
            left.denom().bits(),
            right.numer().bits(),
            right.denom().bits(),
        );
        let (numerator, denominator) = match operation {
            Operation::Add | Operation::Subtract => (
                an.saturating_add(bd)
                    .max(bn.saturating_add(ad))
                    .saturating_add(1),
                ad.saturating_add(bd),
            ),
            Operation::Multiply => (an.saturating_add(bn), ad.saturating_add(bd)),
            Operation::Divide => (an.saturating_add(bd), ad.saturating_add(bn)),
        };
        if numerator > MAX_BITS || denominator > MAX_BITS {
            return Err(limit("arithmetic intermediate coefficient height"));
        }
        self.tick(1)?;
        let value = match operation {
            Operation::Add => left.metered_add(right, self),
            Operation::Subtract => left.metered_sub(right, self),
            Operation::Multiply => left.metered_mul(right, self),
            Operation::Divide => left.metered_div(right, self),
        }
        .map_err(arithmetic)?;
        number_ok(&value, MAX_BITS)?;
        Ok(value)
    }
    fn add(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Add)
    }
    fn subtract(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Subtract)
    }
    fn multiply(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Multiply)
    }
    fn divide(&mut self, left: &BigRational, right: &BigRational) -> Result<BigRational> {
        self.combine(left, right, Operation::Divide)
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
                let (index, maximum) = match dimension {
                    Dimension::ComputeSteps => (0, MAX_STEPS),
                    Dimension::MemoryBytes => (1, MAX_BYTES),
                    Dimension::AllocationCount => (2, MAX_STEPS),
                    _ => continue,
                };
                let remaining = maximum.saturating_sub(next[index]);
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

fn literal(value: &Expr, maximum: u64, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
    work.tick(1)?;
    match value {
        Expr::Integer(value) if value.bits() <= maximum => Ok(()),
        Expr::Integer(_) => Err(limit("integer coefficient height")),
        Expr::Rational(value) => number_ok(value, maximum),
        _ => Err(MatrixError::UnsupportedCertificateDomain),
    }
}

fn admit(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<usize> {
    work.tick(1)?;
    matrix.validate_shape()?;
    if matrix.rows != matrix.cols {
        return Err(MatrixError::NotSquare(matrix.rows, matrix.cols));
    }
    if matrix.rows > MAX_RATIONAL_JORDAN_DIMENSION {
        return Err(limit("matrix dimension exceeds sixteen"));
    }
    // Every borrowed original cell is admitted before shortcuts or copying.
    for entry in &matrix.data {
        literal(entry, MAX_INPUT_BITS, work)?;
    }
    Ok(matrix.rows)
}

fn values(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Vec<BigRational>> {
    let mut result = work.vector(matrix.data.len())?;
    for entry in &matrix.data {
        let limbs = match entry {
            Expr::Integer(value) => value.limb_count() + 1,
            Expr::Rational(value) => value.numer().limb_count() + value.denom().limb_count(),
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        };
        work.storage(1, 64 + 8 * limbs as usize)?;
        result.push(numeric_value(entry).ok_or(MatrixError::UnsupportedCertificateDomain)?);
    }
    Ok(result)
}

fn matrix_from_values(
    n: usize,
    values: Vec<BigRational>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Matrix> {
    let mut entries = work.vector(values.len())?;
    for value in values {
        number_ok(&value, MAX_BITS)?;
        work.tick(1)?;
        if value.is_integer() {
            work.storage(1, 64 + 8 * value.numer().limb_count() as usize)?;
            entries.push(Expr::Integer(value.to_integer()));
        } else {
            entries.push(Expr::Rational(value));
        }
    }
    Matrix::new(n, n, entries)
}

fn matrix_copy(matrix: &Matrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<Matrix> {
    let mut entries = work.vector(matrix.data.len())?;
    for entry in &matrix.data {
        let limbs = match entry {
            Expr::Integer(value) => value.limb_count() + 1,
            Expr::Rational(value) => value.numer().limb_count() + value.denom().limb_count(),
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        };
        work.storage(1, 128 + 8 * limbs as usize)?;
        entries.push(entry.clone());
    }
    Matrix::new(matrix.rows, matrix.cols, entries)
}

fn identity(n: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Vec<BigRational>> {
    let mut values = work.zeros(n * n)?;
    for i in 0..n {
        values[i * n + i] = work.one()?;
    }
    Ok(values)
}

// Generation uses row accumulation; verification below uses separate direct
// coefficient dots and the defining chain relation, never this product helper.
fn product(
    left: &[BigRational],
    right: &[BigRational],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<BigRational>> {
    let mut result = work.zeros(n * n)?;
    for row in 0..n {
        for inner in 0..n {
            work.tick(n + 1)?;
            if left[row * n + inner].is_zero() {
                continue;
            }
            for column in 0..n {
                if !right[inner * n + column].is_zero() {
                    let term = work.multiply(&left[row * n + inner], &right[inner * n + column])?;
                    let at = row * n + column;
                    result[at] = work.add(&result[at], &term)?;
                }
            }
        }
    }
    Ok(result)
}

fn apply(
    matrix: &[BigRational],
    vector: &[BigRational],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<BigRational>> {
    let mut result = work.zeros(n)?;
    for row in 0..n {
        for column in 0..n {
            work.tick(1)?;
            if matrix[row * n + column].is_zero() || vector[column].is_zero() {
                continue;
            }
            let term = work.multiply(&matrix[row * n + column], &vector[column])?;
            result[row] = work.add(&result[row], &term)?;
        }
    }
    Ok(result)
}

fn reduce(
    source: &[BigRational],
    rows: usize,
    columns: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(Vec<BigRational>, Vec<usize>)> {
    let mut entries = work.copies(source)?;
    let mut pivots = work.vector(rows.min(columns))?;
    let mut next = 0;
    for column in 0..columns {
        if next == rows {
            break;
        }
        work.tick(rows - next)?;
        let Some(pivot) = (next..rows).find(|&row| !entries[row * columns + column].is_zero())
        else {
            continue;
        };
        work.tick(columns)?;
        for index in 0..columns {
            entries.swap(next * columns + index, pivot * columns + index);
        }
        let scale = work.copy(&entries[next * columns + column])?;
        for index in column..columns {
            let at = next * columns + index;
            entries[at] = work.divide(&entries[at], &scale)?;
        }
        for row in 0..rows {
            work.tick(1)?;
            if row == next || entries[row * columns + column].is_zero() {
                continue;
            }
            let factor = work.copy(&entries[row * columns + column])?;
            for index in column..columns {
                let term = work.multiply(&factor, &entries[next * columns + index])?;
                let at = row * columns + index;
                entries[at] = work.subtract(&entries[at], &term)?;
            }
        }
        pivots.push(column);
        next += 1;
    }
    Ok((entries, pivots))
}

fn kernel(
    matrix: &[BigRational],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<Vec<BigRational>>> {
    let (reduced, pivots) = reduce(matrix, n, n, work)?;
    let mut basis = work.vector(n - pivots.len())?;
    let zero = work.zero()?;
    for free in 0..n {
        work.tick(n)?;
        if pivots.contains(&free) {
            continue;
        }
        let mut vector = work.zeros(n)?;
        vector[free] = work.one()?;
        for (row, &pivot) in pivots.iter().enumerate() {
            vector[pivot] = work.subtract(&zero, &reduced[row * n + free])?;
        }
        basis.push(vector);
    }
    Ok(basis)
}

/// A generator-only echelon basis of column vectors; dependence is exact.
struct Span {
    pivots: Vec<usize>,
    vectors: Vec<Vec<BigRational>>,
}

impl Span {
    fn new(n: usize, work: &mut Work<'_, impl BudgetMeter>) -> Result<Self> {
        Ok(Self {
            pivots: work.vector(n)?,
            vectors: work.vector(n)?,
        })
    }
    fn insert(
        &mut self,
        vector: &[BigRational],
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<bool> {
        let mut residual = work.copies(vector)?;
        for (&pivot, basis) in self.pivots.iter().zip(&self.vectors) {
            work.tick(1)?;
            if residual[pivot].is_zero() {
                continue;
            }
            let factor = work.copy(&residual[pivot])?;
            for index in pivot..residual.len() {
                let term = work.multiply(&factor, &basis[index])?;
                residual[index] = work.subtract(&residual[index], &term)?;
            }
        }
        work.tick(residual.len())?;
        let Some(pivot) = residual.iter().position(|value| !value.is_zero()) else {
            return Ok(false);
        };
        let scale = work.copy(&residual[pivot])?;
        for value in &mut residual[pivot..] {
            *value = work.divide(value, &scale)?;
        }
        work.tick(self.pivots.len() + 1)?;
        let at = self.pivots.partition_point(|&value| value < pivot);
        self.pivots.insert(at, pivot);
        self.vectors.insert(at, residual);
        Ok(true)
    }
}

fn spectrum_order(
    spectrum: &[(BigRational, usize)],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<usize>> {
    if spectrum.len() > n {
        return Err(invalid("too many spectral hints"));
    }
    let mut total = 0usize;
    for (eigenvalue, multiplicity) in spectrum {
        work.tick(1)?;
        number_ok(eigenvalue, MAX_INPUT_BITS)?;
        if *multiplicity == 0 || *multiplicity > n - total {
            return Err(invalid(
                "spectral multiplicities must be positive and sum to the dimension",
            ));
        }
        total += multiplicity;
    }
    if total != n {
        return Err(invalid("spectral multiplicities must sum to the dimension"));
    }
    let mut indices: Vec<usize> = work.vector(spectrum.len())?;
    for index in 0..spectrum.len() {
        let mut at = indices.len();
        while at > 0 {
            match work.compare(&spectrum[indices[at - 1]].0, &spectrum[index].0)? {
                Ordering::Less => break,
                Ordering::Equal => return Err(invalid("spectral hints repeat an eigenvalue")),
                Ordering::Greater => at -= 1,
            }
        }
        work.tick(indices.len() + 1)?;
        indices.insert(at, index);
    }
    Ok(indices)
}

fn generate(
    matrix: &Matrix,
    spectrum: &[(BigRational, usize)],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalJordanCandidate> {
    let n = admit(matrix, work)?;
    let order = spectrum_order(spectrum, n, work)?;
    let original = values(matrix, work)?;
    let mut columns = work.vector(n)?;
    let mut blocks = work.vector(n)?;

    for index in order {
        let (eigenvalue, multiplicity) = &spectrum[index];
        let mut shifted = work.copies(&original)?;
        for diagonal in 0..n {
            let at = diagonal * n + diagonal;
            shifted[at] = work.subtract(&shifted[at], eigenvalue)?;
        }
        let mut power = identity(n, work)?;
        let mut kernels = work.vector(multiplicity + 2)?;
        kernels.push(work.vector::<Vec<BigRational>>(0)?);
        for _ in 0..*multiplicity {
            power = product(&shifted, &power, n, work)?;
            let basis = kernel(&power, n, work)?;
            let dimension = basis.len();
            if dimension > *multiplicity || dimension == kernels.last().expect("K_0 retained").len()
            {
                return Err(invalid(
                    "spectral hint has the wrong algebraic multiplicity",
                ));
            }
            kernels.push(basis);
            if dimension == *multiplicity {
                // Check stabilization as well as reaching the claimed size.
                // This rejects an understated multiplicity of a longer chain.
                power = product(&shifted, &power, n, work)?;
                let next = kernel(&power, n, work)?;
                if next.len() != *multiplicity {
                    return Err(invalid("spectral hint omits generalized eigenvectors"));
                }
                kernels.push(next);
                break;
            }
        }
        if kernels.len() < 3 || kernels[kernels.len() - 2].len() != *multiplicity {
            return Err(invalid(
                "spectral hint is not a complete rational eigenspace",
            ));
        }
        let largest = kernels.len() - 2;
        for size in (1..=largest).rev() {
            let current = kernels[size].len();
            let smaller = kernels[size - 1].len();
            let larger = kernels[size + 1].len();
            let at_least = current
                .checked_sub(smaller)
                .ok_or_else(|| invalid("kernel dimensions decreased"))?;
            let above = larger
                .checked_sub(current)
                .ok_or_else(|| invalid("kernel dimensions decreased"))?;
            let needed = at_least
                .checked_sub(above)
                .ok_or_else(|| invalid("kernel growth is not concave"))?;
            if needed == 0 {
                continue;
            }
            let mut span = Span::new(n, work)?;
            for vector in &kernels[size - 1] {
                span.insert(vector, work)?;
            }
            for vector in &kernels[size + 1] {
                let image = apply(&shifted, vector, n, work)?;
                span.insert(&image, work)?;
            }
            let mut chosen = 0;
            for top in &kernels[size] {
                if !span.insert(top, work)? {
                    continue;
                }
                if columns.len() + size > n || blocks.len() == n {
                    return Err(invalid("Jordan chains exceed the original dimension"));
                }
                let mut chain = work.vector(size)?;
                chain.push(work.copies(top)?);
                for _ in 1..size {
                    let next = apply(&shifted, chain.last().expect("chain top retained"), n, work)?;
                    chain.push(next);
                }
                work.tick(size)?;
                columns.extend(chain.into_iter().rev());
                blocks.push(RationalJordanBlock {
                    eigenvalue: work.copy(eigenvalue)?,
                    size,
                });
                chosen += 1;
                if chosen == needed {
                    break;
                }
            }
            if chosen != needed {
                return Err(invalid(
                    "failed to construct a complete Jordan quotient basis",
                ));
            }
        }
    }
    if columns.len() != n {
        return Err(invalid("Jordan chains do not span the original dimension"));
    }
    let mut transform = work.vector(n * n)?;
    let mut augmented = work.vector(n * 2 * n)?;
    for row in 0..n {
        for column in &columns {
            transform.push(work.copy(&column[row])?);
            augmented.push(work.copy(&column[row])?);
        }
        for column in 0..n {
            augmented.push(if row == column {
                work.one()?
            } else {
                work.zero()?
            });
        }
    }
    let (reduced, pivots) = reduce(&augmented, n, 2 * n, work)?;
    if pivots.len() != n
        || pivots
            .iter()
            .enumerate()
            .any(|(index, &pivot)| index != pivot)
    {
        return Err(invalid("generated Jordan transform is singular"));
    }
    let mut inverse = work.vector(n * n)?;
    for row in 0..n {
        for column in 0..n {
            inverse.push(work.copy(&reduced[row * 2 * n + n + column])?);
        }
    }
    let transform = matrix_from_values(n, transform, work)?;
    let inverse = matrix_from_values(n, inverse, work)?;
    work.tick(1)?;
    Ok(RationalJordanCandidate {
        blocks,
        transform,
        inverse,
    })
}

fn reference_dot<'a>(
    pairs: impl Iterator<Item = (&'a BigRational, &'a BigRational)>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let mut result = work.zero()?;
    for (left, right) in pairs {
        let term = work.multiply(left, right)?;
        result = work.add(&result, &term)?;
    }
    Ok(result)
}

fn verify(
    matrix: &Matrix,
    candidate: &RationalJordanCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedRationalJordanDecomposition> {
    let n = admit(matrix, work)?;
    if candidate.blocks.len() > n {
        return Err(invalid("too many Jordan blocks"));
    }
    let mut total = 0;
    for (index, block) in candidate.blocks.iter().enumerate() {
        work.tick(1)?;
        number_ok(&block.eigenvalue, MAX_BITS)?;
        if block.size == 0 || block.size > n - total {
            return Err(invalid(
                "Jordan block dimensions must be positive and total the source dimension",
            ));
        }
        total += block.size;
        if index > 0 {
            let previous = &candidate.blocks[index - 1];
            match work.compare(&previous.eigenvalue, &block.eigenvalue)? {
                Ordering::Greater => return Err(invalid("Jordan eigenvalues are not ordered")),
                Ordering::Equal if previous.size < block.size => {
                    return Err(invalid(
                        "equal-eigenvalue Jordan blocks are not decreasing in size",
                    ));
                }
                _ => {}
            }
        }
    }
    if total != n {
        return Err(invalid("Jordan blocks omit source dimensions"));
    }
    for witness in [&candidate.transform, &candidate.inverse] {
        witness.validate_shape()?;
        if witness.rows != n || witness.cols != n {
            return Err(invalid("similarity witnesses have the wrong shape"));
        }
        for entry in &witness.data {
            literal(entry, MAX_BITS, work)?;
        }
    }
    let original = values(matrix, work)?;
    let transform = values(&candidate.transform, work)?;
    let inverse = values(&candidate.inverse, work)?;

    // A direct left-inverse identity establishes complete column independence.
    // This checker never calls the generator's elimination or rank routines.
    for row in 0..n {
        for column in 0..n {
            let value = reference_dot(
                (0..n).map(|inner| (&inverse[row * n + inner], &transform[inner * n + column])),
                work,
            )?;
            if (row == column && !value.is_one()) || (row != column && !value.is_zero()) {
                return Err(invalid("transform has no certified left inverse"));
            }
        }
    }

    // Check each defining chain equation against the original A. J is built
    // only after acceptance, so an arbitrary supplied matrix cannot masquerade
    // as Jordan normal form while satisfying a generic similarity identity.
    let mut offset = 0;
    for block in &candidate.blocks {
        for local in 0..block.size {
            let column = offset + local;
            for row in 0..n {
                let left = reference_dot(
                    (0..n)
                        .map(|inner| (&original[row * n + inner], &transform[inner * n + column])),
                    work,
                )?;
                let mut right = work.multiply(&block.eigenvalue, &transform[row * n + column])?;
                if local > 0 {
                    right = work.add(&right, &transform[row * n + column - 1])?;
                }
                if work.compare(&left, &right)? != Ordering::Equal {
                    return Err(invalid(
                        "Jordan chains do not intertwine the original matrix",
                    ));
                }
            }
        }
        offset += block.size;
    }

    let mut jordan = work.zeros(n * n)?;
    let mut blocks = work.vector(candidate.blocks.len())?;
    offset = 0;
    for block in &candidate.blocks {
        for local in 0..block.size {
            let diagonal = offset + local;
            jordan[diagonal * n + diagonal] = work.copy(&block.eigenvalue)?;
            if local + 1 < block.size {
                jordan[diagonal * n + diagonal + 1] = work.one()?;
            }
        }
        blocks.push(RationalJordanBlock {
            eigenvalue: work.copy(&block.eigenvalue)?,
            size: block.size,
        });
        offset += block.size;
    }
    let jordan = matrix_from_values(n, jordan, work)?;
    let source = matrix_copy(matrix, work)?;
    let transform = matrix_copy(&candidate.transform, work)?;
    let inverse = matrix_copy(&candidate.inverse, work)?;
    work.tick(1)?;
    Ok(VerifiedRationalJordanDecomposition {
        matrix: source,
        certificate: RationalJordanCandidate {
            blocks,
            transform,
            inverse,
        },
        jordan,
    })
}

/// Independently check complete similarity against the original source.
/// No eigensolver, inverse search, rank computation or kernel generator is used.
pub fn verify_rational_jordan_candidate(
    matrix: &Matrix,
    candidate: &RationalJordanCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalJordanDecomposition> {
    verify(matrix, candidate, &mut Work::new(verifier))
}

impl Matrix {
    /// Complete rational Jordan form. Spectrum entries are distinct untrusted
    /// eigenvalue/multiplicity hints; their total must equal the dimension.
    pub fn rational_jordan(
        &self,
        spectrum: &[(BigRational, usize)],
    ) -> Result<VerifiedRationalJordanDecomposition> {
        self.metered_rational_jordan(spectrum, &mut Unbounded, &mut Unbounded)
    }

    /// Separate phase meters remain cumulative through admission/publication.
    /// All failures after entry are terminal; no generic algebra retry occurs.
    pub fn metered_rational_jordan(
        &self,
        spectrum: &[(BigRational, usize)],
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedRationalJordanDecomposition> {
        let mut generation = Work::new(generator);
        let mut verification = Work::new(verifier);
        let candidate = generate(self, spectrum, &mut generation)?;
        let result = verify(self, &candidate, &mut verification)?;
        generation.checkpoint().map_err(stopped)?;
        verification.checkpoint().map_err(stopped)?;
        Ok(result)
    }

    /// Generate untrusted evidence for a separate checker or mutation test.
    pub fn rational_jordan_candidate_with_meter(
        &self,
        spectrum: &[(BigRational, usize)],
        generator: &mut impl BudgetMeter,
    ) -> Result<RationalJordanCandidate> {
        generate(self, spectrum, &mut Work::new(generator))
    }
}
