//! Pivoted exact block-LDL generation for rational quadratic-form inertia.
//!
//! The input may be nonsymmetric: its quadratic form is governed by A+A^T.
//! A nonzero scalar diagonal supplies a 1x1 pivot. When every remaining
//! diagonal vanishes, a nonzero off-diagonal supplies an invertible 2x2 pivot.
//! This covers singular and indefinite forms without numerical eigenvalues,
//! square roots, or exponentially many principal minors. An independent
//! fsym-proof-kernel checker accepts the complete congruence, not the pivots.

use crate::{Matrix, MatrixError, SparseMatrix};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use fsym_proof_kernel::rational_inertia::{
    MAX_RATIONAL_INERTIA_DIMENSION, RATIONAL_INERTIA_SCHEMA_VERSION, RationalInertiaError,
    RationalInertiaReceipt, RationalInertiaWitness, verify_rational_inertia,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry, SparseRrefError};
use num_traits::{One, Zero};

const MAX_INPUT_BITS: u64 = 4096;
const MAX_BITS: u64 = 8192;
const MAX_STEPS: u64 = 20_000_000;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
type Result<T> = std::result::Result<T, MatrixError>;

/// Untrusted portable source and block-LDL witness. The source is a separate
/// input to the proof kernel so receivers can bind it to their original matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalInertiaCandidate {
    pub source: RationalSparseMatrix,
    pub witness: RationalInertiaWitness,
}

/// Original source, complete congruence and independent acceptance receipt.
#[derive(Debug)]
pub struct VerifiedRationalInertia {
    source: RationalSparseMatrix,
    witness: RationalInertiaWitness,
    receipt: RationalInertiaReceipt,
}
impl VerifiedRationalInertia {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn witness(&self) -> &RationalInertiaWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &RationalInertiaReceipt {
        &self.receipt
    }
}

fn limit(message: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("rational matrix inertia: {message}"))
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
        error => limit(&error.to_string()),
    }
}
fn verification(error: RationalInertiaError) -> MatrixError {
    match error {
        RationalInertiaError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        RationalInertiaError::Reference(SparseRrefError::Arithmetic(error)) => arithmetic(error),
        RationalInertiaError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}
fn coefficient(value: &BigRational, bits: u64) -> Result<()> {
    if !value.denom().is_positive() {
        return Err(MatrixError::InvalidCertificate(
            "invalid rational denominator".into(),
        ));
    }
    if value.numer().bits().max(value.denom().bits()) > bits {
        return Err(limit("coefficient height"));
    }
    Ok(())
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
    fn tick(&mut self, steps: usize) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, steps.max(1) as u64)
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
    fn vector<T>(&mut self, n: usize) -> Result<Vec<T>> {
        self.storage(
            n.checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| limit("container size overflow"))?,
        )?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(n)
            .map_err(|_| limit("container allocation"))?;
        Ok(result)
    }
    fn zeros(&mut self, n: usize) -> Result<Vec<BigRational>> {
        let mut values = self.vector(n)?;
        self.storage(n.saturating_mul(64))?;
        values.resize_with(n, BigRational::zero);
        Ok(values)
    }
    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        coefficient(value, MAX_BITS)?;
        let limbs = value.numer().limb_count() + value.denom().limb_count();
        self.tick(limbs as usize)?;
        self.storage(64 + 8 * limbs as usize)?;
        Ok(value.clone())
    }
    fn scalar(&mut self, value: &Expr) -> Result<BigRational> {
        match value {
            Expr::Integer(value) => {
                self.storage(64 + 8 * value.limb_count() as usize)?;
                Ok(BigRational::from_integer(value.clone()))
            }
            Expr::Rational(value) => self.copy(value),
            _ => Err(MatrixError::UnsupportedCertificateDomain),
        }
    }
    fn combine(
        &mut self,
        left: &BigRational,
        right: &BigRational,
        operation: u8,
    ) -> Result<BigRational> {
        coefficient(left, MAX_BITS)?;
        coefficient(right, MAX_BITS)?;
        let (an, ad, bn, bd) = (
            left.numer().bits(),
            left.denom().bits(),
            right.numer().bits(),
            right.denom().bits(),
        );
        let (numerator, denominator) = match operation {
            0 | 1 => ((an + bd).max(bn + ad) + 1, ad + bd),
            2 => (an + bn, ad + bd),
            _ => (an + bd, ad + bn),
        };
        if numerator > MAX_BITS || denominator > MAX_BITS {
            return Err(limit("arithmetic intermediate height"));
        }
        self.tick(1)?;
        let value = match operation {
            0 => left.metered_add(right, self),
            1 => left.metered_sub(right, self),
            2 => left.metered_mul(right, self),
            _ => left.metered_div(right, self),
        }
        .map_err(arithmetic)?;
        coefficient(&value, MAX_BITS)?;
        Ok(value)
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.combine(a, b, 0)
    }
    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.combine(a, b, 1)
    }
    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.combine(a, b, 2)
    }
    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.combine(a, b, 3)
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
            let (i, cap) = match dimension {
                Dimension::ComputeSteps => (0, MAX_STEPS),
                Dimension::MemoryBytes => (1, MAX_BYTES),
                Dimension::AllocationCount => (2, MAX_STEPS),
                _ => continue,
            };
            let remaining = cap.saturating_sub(next[i]);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            next[i] += amount;
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
    if rows != columns {
        return Err(MatrixError::NotSquare(rows, columns));
    }
    if rows > MAX_RATIONAL_INERTIA_DIMENSION {
        return Err(limit("dimension exceeds 64"));
    }
    let mut count = 0;
    let mut previous = None;
    for (coordinate, value) in entries {
        work.tick(1)?;
        count += 1;
        if count > rows * rows
            || coordinate.0 >= rows
            || coordinate.1 >= rows
            || previous.is_some_and(|p| p >= coordinate)
        {
            return Err(MatrixError::InvalidCertificate(
                "invalid matrix source coordinates".into(),
            ));
        }
        previous = Some(coordinate);
        match value {
            Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {}
            Expr::Integer(_) => return Err(limit("input coefficient exceeds 4096 bits")),
            Expr::Rational(value) => coefficient(value, MAX_INPUT_BITS)?,
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        }
    }
    Ok(count)
}

fn snapshot<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = ((usize, usize), &'a Expr)> + Clone,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    let count = admit(rows, columns, entries.clone(), work)?;
    let mut values = work.vector(count)?;
    for ((row, column), value) in entries {
        work.tick(1)?;
        if !value.is_zero() {
            values.push(SparseRationalEntry {
                row,
                column,
                value: work.scalar(value)?,
            });
        }
    }
    Ok(RationalSparseMatrix {
        rows,
        columns,
        entries: values,
    })
}

fn sparse(
    values: &[BigRational],
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    let mut entries = work.vector(values.len())?;
    for (index, value) in values.iter().enumerate() {
        work.tick(1)?;
        if !value.is_zero() {
            entries.push(SparseRationalEntry {
                row: index / n,
                column: index % n,
                value: work.copy(value)?,
            });
        }
    }
    Ok(RationalSparseMatrix {
        rows: n,
        columns: n,
        entries,
    })
}

struct Decomposition {
    n: usize,
    reduced: Vec<BigRational>,
    lower: Vec<BigRational>,
    diagonal: Vec<BigRational>,
    permutation: Vec<usize>,
}
impl Decomposition {
    fn swap(
        &mut self,
        a: usize,
        b: usize,
        completed: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        if a == b {
            return Ok(());
        }
        work.tick(2 * self.n + completed + 1)?;
        for i in 0..self.n {
            self.reduced.swap(a * self.n + i, b * self.n + i);
        }
        for i in 0..self.n {
            self.reduced.swap(i * self.n + a, i * self.n + b);
        }
        for i in 0..completed {
            self.lower.swap(a * self.n + i, b * self.n + i);
        }
        self.permutation.swap(a, b);
        Ok(())
    }
}

fn generate(
    source: RationalSparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalInertiaCandidate> {
    let n = source.rows;
    let mut a = work.zeros(n * n)?;
    for entry in &source.entries {
        a[entry.row * n + entry.column] = work.copy(&entry.value)?;
    }
    let mut reduced = work.zeros(n * n)?;
    for i in 0..n {
        for j in 0..=i {
            let value = work.add(&a[i * n + j], &a[j * n + i])?;
            reduced[j * n + i] = work.copy(&value)?;
            reduced[i * n + j] = value;
        }
    }
    let mut lower = work.zeros(n * n)?;
    work.storage(n * 64)?;
    for i in 0..n {
        lower[i * n + i] = BigRational::one();
    }
    let diagonal = work.zeros(n * n)?;
    let mut permutation = work.vector(n)?;
    permutation.extend(0..n);
    let mut d = Decomposition {
        n,
        reduced,
        lower,
        diagonal,
        permutation,
    };
    let mut k = 0;
    while k < n {
        work.tick(1)?;
        let mut pivot = None;
        for i in k..n {
            work.tick(1)?;
            if !d.reduced[i * n + i].is_zero() {
                pivot = Some(i);
                break;
            }
        }
        if let Some(pivot) = pivot {
            d.swap(k, pivot, k, work)?;
            let value = work.copy(&d.reduced[k * n + k])?;
            d.diagonal[k * n + k] = work.copy(&value)?;
            for i in k + 1..n {
                d.lower[i * n + k] = work.div(&d.reduced[i * n + k], &value)?;
            }
            for i in k + 1..n {
                for j in k + 1..=i {
                    let correction = work.mul(&d.lower[i * n + k], &d.reduced[k * n + j])?;
                    let value = work.sub(&d.reduced[i * n + j], &correction)?;
                    d.reduced[j * n + i] = work.copy(&value)?;
                    d.reduced[i * n + j] = value;
                }
            }
            k += 1;
        } else {
            let mut pair = None;
            'search: for i in k..n {
                for j in i + 1..n {
                    work.tick(1)?;
                    if !d.reduced[i * n + j].is_zero() {
                        pair = Some((i, j));
                        break 'search;
                    }
                }
            }
            let Some((i, j)) = pair else {
                break;
            };
            d.swap(k, i, k, work)?;
            d.swap(k + 1, j, k, work)?;
            let value = work.copy(&d.reduced[k * n + k + 1])?;
            d.diagonal[k * n + k + 1] = work.copy(&value)?;
            d.diagonal[(k + 1) * n + k] = work.copy(&value)?;
            for i in k + 2..n {
                d.lower[i * n + k] = work.div(&d.reduced[i * n + k + 1], &value)?;
                d.lower[i * n + k + 1] = work.div(&d.reduced[i * n + k], &value)?;
            }
            for i in k + 2..n {
                for j in k + 2..=i {
                    let first = work.mul(&d.lower[i * n + k], &d.reduced[k * n + j])?;
                    let second = work.mul(&d.lower[i * n + k + 1], &d.reduced[(k + 1) * n + j])?;
                    let correction = work.add(&first, &second)?;
                    let value = work.sub(&d.reduced[i * n + j], &correction)?;
                    d.reduced[j * n + i] = work.copy(&value)?;
                    d.reduced[i * n + j] = value;
                }
            }
            k += 2;
        }
    }
    let witness = RationalInertiaWitness {
        schema_version: RATIONAL_INERTIA_SCHEMA_VERSION,
        permutation: d.permutation,
        lower: sparse(&d.lower, n, work)?,
        diagonal: sparse(&d.diagonal, n, work)?,
    };
    work.tick(1)?;
    Ok(RationalInertiaCandidate { source, witness })
}

fn accept(
    candidate: RationalInertiaCandidate,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalInertia> {
    generator.checkpoint().map_err(stopped)?;
    let receipt = verify_rational_inertia(&candidate.source, &candidate.witness, verifier)
        .map_err(verification)?;
    generator.checkpoint().map_err(stopped)?;
    verifier.checkpoint().map_err(stopped)?;
    Ok(VerifiedRationalInertia {
        source: candidate.source,
        witness: candidate.witness,
        receipt,
    })
}

impl Matrix {
    /// Borrowed input preflight for bridges that validate before copying cells.
    pub fn validate_rational_inertia_source(&self) -> Result<()> {
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
    /// Complete positive/negative/zero inertia of the rational quadratic form.
    pub fn inertia(&self) -> Result<VerifiedRationalInertia> {
        self.inertia_with_meters(&mut Unbounded, &mut Unbounded)
    }
    pub fn inertia_with_meters(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedRationalInertia> {
        let candidate = self.inertia_candidate_with_meter(generator)?;
        accept(candidate, generator, verifier)
    }
    pub fn inertia_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<RationalInertiaCandidate> {
        let mut work = Work::new(generator);
        work.tick(1)?;
        self.validate_shape()?;
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
    /// Sparse admission visits stored entries before any dense materialization.
    pub fn validate_rational_inertia_source(&self) -> Result<()> {
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
    /// Bounded sparse-source admission before any dense factorization storage.
    pub fn inertia(&self) -> Result<VerifiedRationalInertia> {
        self.inertia_with_meters(&mut Unbounded, &mut Unbounded)
    }
    pub fn inertia_with_meters(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedRationalInertia> {
        let candidate = self.inertia_candidate_with_meter(generator)?;
        accept(candidate, generator, verifier)
    }
    pub fn inertia_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<RationalInertiaCandidate> {
        let mut work = Work::new(generator);
        let source = snapshot(
            self.rows(),
            self.cols(),
            self.entries()
                .iter()
                .map(|(&coordinate, value)| (coordinate, value)),
            &mut work,
        )?;
        generate(source, &mut work)
    }
}
