//! Independent exact sparse row-reduction and complete affine certificates over Q.
//!
//! Acceptance replays only invertible elementary row operations and checks
//! every reduced-row-echelon invariant. In particular, a list of plausible
//! pivots or null vectors cannot establish rank. The checker has no dependency
//! on a pivot-search/elimination generator and never allocates rows * columns
//! cells. Its simple reference representation is sorted sparse row vectors.
//! Affine acceptance additionally checks original-source particular/kernel
//! residuals and canonical free coordinates for every right-hand side.
//!
//! All replay, coefficient copying, hashing and receipt publication share the
//! caller's meter, with additional cumulative local limits. Cancellation or a
//! limit returns no receipt. This synchronous checker does not itself reserve
//! verifier resources; its caller must supply the protected meter when required.

use crate::linear_system::LinearRowOperation;
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::{fmt, marker::PhantomData};
use thiserror::Error;

pub const SPARSE_RREF_SCHEMA_VERSION: u32 = 1;
pub const SPARSE_AFFINE_SCHEMA_VERSION: u32 = 1;
pub const MAX_SPARSE_RREF_DIMENSION: usize = 8_192;
pub const MAX_SPARSE_RREF_ENTRIES: usize = 262_144;
pub const MAX_SPARSE_RREF_OPERATIONS: usize = 1_000_000;
pub const MAX_SPARSE_RREF_COEFFICIENT_BITS: u64 = 8_192;
const MAX_WORK: u64 = 20_000_000;
const MAX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;

/// One nonzero rational cell. Coordinates are relative to the stated shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseRationalEntry {
    pub row: usize,
    pub column: usize,
    #[serde(deserialize_with = "bounded_scalar")]
    pub value: BigRational,
}

/// Canonical sparse claim data: strictly increasing (row, column) coordinates,
/// without duplicates, explicit zeros or out-of-shape entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalSparseMatrix {
    pub rows: usize,
    pub columns: usize,
    #[serde(deserialize_with = "bounded_entries")]
    pub entries: Vec<SparseRationalEntry>,
}

/// Untrusted portable witness. The receiver supplies the expected source
/// matrix separately; neither a stored digest nor this witness grants evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseRrefWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_operations")]
    pub operations: Vec<LinearRowOperation>,
    pub reduced: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_pivots")]
    pub pivots: Vec<usize>,
}

/// Untrusted classification of the complete matrix equation `A * X = B`.
/// A solvable system is represented canonically as `X = particular + kernel*T`,
/// where the rows of `T` correspond to ascending free coefficient columns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SparseAffineOutcome {
    /// A checked augmented RREF row with zero coefficient entries and a
    /// nonzero right-hand side establishes that the whole system is inconsistent.
    Inconsistent { row: usize },
    Solvable {
        particular: RationalSparseMatrix,
        kernel: RationalSparseMatrix,
        #[serde(deserialize_with = "bounded_pivots")]
        free_columns: Vec<usize>,
    },
}

/// Portable proposal for all right-hand sides together. The expected original
/// coefficient and right-hand-side matrices are separate verifier inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseAffineWitness {
    pub schema_version: u32,
    pub reduction: SparseRrefWitness,
    pub outcome: SparseAffineOutcome,
}

/// A checked source-to-RREF relation and its exact rank. Retain the source and
/// witness with this receipt; digests are commitments, not mathematical proofs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseRrefReceipt {
    rows: usize,
    columns: usize,
    pivots: Vec<usize>,
    source_digest: [u8; 32],
    reduced_digest: [u8; 32],
}

impl SparseRrefReceipt {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn reduced_digest(&self) -> [u8; 32] {
        self.reduced_digest
    }
}

/// Accepted complete affine classification, bound to the original `A`, `B`,
/// and canonical output matrices. No unchecked constructor or deserializer
/// exists. Digests are commitments; retain the inputs and witness for replay.
///
/// `rank` and `free_columns` describe the coefficient matrix even when the
/// system is inconsistent. With zero RHS columns the parameter matrix has
/// shape `nullity x 0`: a nonzero kernel does not imply multiple matrix values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseAffineReceipt {
    rows: usize,
    columns: usize,
    rhs_columns: usize,
    rank: usize,
    augmented_rank: usize,
    free_columns: Vec<usize>,
    contradiction_row: Option<usize>,
    source_digest: [u8; 32],
    rhs_digest: [u8; 32],
    particular_digest: Option<[u8; 32]>,
    kernel_digest: Option<[u8; 32]>,
}

impl SparseAffineReceipt {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rhs_columns(&self) -> usize {
        self.rhs_columns
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn augmented_rank(&self) -> usize {
        self.augmented_rank
    }
    pub fn free_columns(&self) -> &[usize] {
        &self.free_columns
    }
    pub fn is_consistent(&self) -> bool {
        self.contradiction_row.is_none()
    }
    pub fn contradiction_row(&self) -> Option<usize> {
        self.contradiction_row
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn rhs_digest(&self) -> [u8; 32] {
        self.rhs_digest
    }
    pub fn particular_digest(&self) -> Option<[u8; 32]> {
        self.particular_digest
    }
    pub fn kernel_digest(&self) -> Option<[u8; 32]> {
        self.kernel_digest
    }
}

#[derive(Debug, Error)]
pub enum SparseRrefError {
    #[error("sparse RREF certificate rejected: {0}")]
    Invalid(&'static str),
    #[error("sparse RREF verification resource limit: {0}")]
    Limit(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for SparseRrefError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, SparseRrefError>;
type Row = Vec<(usize, BigRational)>;

fn scalar_bytes(value: &BigRational) -> u64 {
    (value.numer().limb_count() + value.denom().limb_count()).max(1) * 8
}

// Bound each integer before rational normalization. BigRational's ordinary
// decoder is canonical, but admits a wider foundational coefficient universe.
struct BoundedInteger(BigInt);
impl<'de> Deserialize<'de> for BoundedInteger {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let value = BigInt::deserialize(d)?;
        if value.bits() > MAX_SPARSE_RREF_COEFFICIENT_BITS {
            return Err(serde::de::Error::custom("sparse RREF coefficient height"));
        }
        Ok(Self(value))
    }
}

pub(super) fn bounded_scalar<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<BigRational, D::Error> {
    let (BoundedInteger(numerator), BoundedInteger(denominator)) =
        <(BoundedInteger, BoundedInteger)>::deserialize(d)?;
    if !denominator.is_positive() || !numerator.gcd(&denominator).is_one() {
        return Err(serde::de::Error::custom(
            "noncanonical sparse RREF rational",
        ));
    }
    Ok(BigRational::new(numerator, denominator))
}

trait WireSize {
    fn owned_bytes(&self) -> u64;
}
impl WireSize for SparseRationalEntry {
    fn owned_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as u64 + scalar_bytes(&self.value)
    }
}
impl WireSize for LinearRowOperation {
    fn owned_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as u64
            + match self {
                Self::Swap { .. } => 0,
                Self::Scale { factor, .. } | Self::Add { factor, .. } => scalar_bytes(factor),
            }
    }
}
impl WireSize for usize {
    fn owned_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as u64
    }
}

fn bounded_vec<'de, D, T, const MAX: usize>(d: D) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + WireSize,
{
    struct Bounded<T, const MAX: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de> + WireSize, const MAX: usize> Visitor<'de> for Bounded<T, MAX> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "at most {MAX} bounded sparse certificate items")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<T>, A::Error> {
            if seq.size_hint().is_some_and(|size| size > MAX) {
                return Err(serde::de::Error::custom("sparse RREF sequence limit"));
            }
            let mut values = Vec::new();
            let mut bytes = 0u64;
            loop {
                if values.len() == MAX {
                    if seq.next_element::<IgnoredAny>()?.is_some() {
                        return Err(serde::de::Error::custom("sparse RREF sequence limit"));
                    }
                    return Ok(values);
                }
                let Some(value) = seq.next_element::<T>()? else {
                    return Ok(values);
                };
                bytes = bytes.saturating_add(value.owned_bytes());
                if bytes > MAX_ALLOCATION_BYTES {
                    return Err(serde::de::Error::custom(
                        "sparse RREF wire allocation limit",
                    ));
                }
                values.push(value);
            }
        }
    }
    d.deserialize_seq(Bounded::<T, MAX>(PhantomData))
}

fn bounded_entries<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<SparseRationalEntry>, D::Error> {
    bounded_vec::<D, SparseRationalEntry, MAX_SPARSE_RREF_ENTRIES>(d)
}
fn bounded_operations<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<LinearRowOperation>, D::Error> {
    bounded_vec::<D, LinearRowOperation, MAX_SPARSE_RREF_OPERATIONS>(d)
}
fn bounded_pivots<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<usize>, D::Error> {
    bounded_vec::<D, usize, MAX_SPARSE_RREF_DIMENSION>(d)
}

pub(super) fn check_scalar(value: &BigRational) -> Result<()> {
    if !value.denom().is_positive() {
        return Err(SparseRrefError::Invalid("invalid rational denominator"));
    }
    if value.numer().bits().max(value.denom().bits()) > MAX_SPARSE_RREF_COEFFICIENT_BITS {
        return Err(SparseRrefError::Limit("coefficient height"));
    }
    Ok(())
}

pub(super) struct Reference<'a, M> {
    meter: &'a mut M,
    steps: u64,
    bytes: u64,
}
impl<M: BudgetMeter> BudgetMeter for Reference<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let (mut steps, mut bytes) = (self.steps, self.bytes);
        for &(dimension, amount) in charges {
            let (used, cap) = match dimension {
                Dimension::ComputeSteps => (&mut steps, MAX_WORK),
                Dimension::MemoryBytes => (&mut bytes, MAX_ALLOCATION_BYTES),
                _ => continue,
            };
            let remaining = cap.saturating_sub(*used);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *used += amount;
        }
        self.meter.charge_batch(charges)?;
        self.steps = steps;
        self.bytes = bytes;
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

impl<'a, M: BudgetMeter> Reference<'a, M> {
    pub(super) fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            steps: 0,
            bytes: 0,
        }
    }
    pub(super) fn tick(&mut self, steps: u64) -> Result<()> {
        self.checkpoint()?;
        self.charge(Dimension::ComputeSteps, steps.max(1))?;
        Ok(())
    }
    fn storage(&mut self, bytes: u64) -> Result<()> {
        self.tick(1)?;
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes.max(1)),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>> {
        self.storage(capacity.saturating_mul(std::mem::size_of::<T>()) as u64)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(capacity)
            .map_err(|_| SparseRrefError::Limit("container allocation"))?;
        Ok(result)
    }
    pub(super) fn clone_value(&mut self, value: &BigRational) -> Result<BigRational> {
        self.storage(scalar_bytes(value))?;
        Ok(value.clone())
    }
    pub(super) fn multiply(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        if a.numer().bits().saturating_add(b.numer().bits()) > MAX_SPARSE_RREF_COEFFICIENT_BITS
            || a.denom().bits().saturating_add(b.denom().bits()) > MAX_SPARSE_RREF_COEFFICIENT_BITS
        {
            return Err(SparseRrefError::Limit("multiplication intermediate"));
        }
        self.tick(1)?;
        let result = a.metered_mul(b, self)?;
        check_scalar(&result)?;
        Ok(result)
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        let numerator = a
            .numer()
            .bits()
            .saturating_add(b.denom().bits())
            .max(b.numer().bits().saturating_add(a.denom().bits()))
            .saturating_add(1);
        if numerator > MAX_SPARSE_RREF_COEFFICIENT_BITS
            || a.denom().bits().saturating_add(b.denom().bits()) > MAX_SPARSE_RREF_COEFFICIENT_BITS
        {
            return Err(SparseRrefError::Limit("addition intermediate"));
        }
        self.tick(1)?;
        let result = a.metered_add(b, self)?;
        check_scalar(&result)?;
        Ok(result)
    }
    fn check_matrix(&mut self, matrix: &RationalSparseMatrix) -> Result<()> {
        self.tick(1)?;
        if matrix.rows > MAX_SPARSE_RREF_DIMENSION || matrix.columns > MAX_SPARSE_RREF_DIMENSION {
            return Err(SparseRrefError::Limit("matrix dimensions"));
        }
        if matrix.entries.len() > MAX_SPARSE_RREF_ENTRIES {
            return Err(SparseRrefError::Limit("stored entries"));
        }
        let mut previous = None;
        for entry in &matrix.entries {
            self.tick(1)?;
            check_scalar(&entry.value)?;
            let coordinate = (entry.row, entry.column);
            if entry.row >= matrix.rows
                || entry.column >= matrix.columns
                || previous.is_some_and(|last| last >= coordinate)
                || entry.value.is_zero()
            {
                return Err(SparseRrefError::Invalid("noncanonical sparse coordinates"));
            }
            previous = Some(coordinate);
        }
        Ok(())
    }
    fn rows(&mut self, matrix: &RationalSparseMatrix) -> Result<Vec<Row>> {
        let mut rows = self.vector(matrix.rows)?;
        let mut entries = matrix.entries.as_slice();
        for index in 0..matrix.rows {
            self.tick(1)?;
            let length = entries.partition_point(|entry| entry.row == index);
            let mut row = self.vector(length)?;
            for entry in &entries[..length] {
                row.push((entry.column, self.clone_value(&entry.value)?));
            }
            entries = &entries[length..];
            rows.push(row);
        }
        Ok(rows)
    }
    fn add_row(
        &mut self,
        target: &Row,
        source: &Row,
        factor: &BigRational,
        allowance: usize,
    ) -> Result<Row> {
        let capacity = (target.len() + source.len()).min(allowance);
        let mut result = self.vector(capacity)?;
        let (mut i, mut j) = (0, 0);
        while i < target.len() || j < source.len() {
            self.tick(1)?;
            let (column, value) = match (target.get(i), source.get(j)) {
                (Some((a, av)), Some((b, bv))) if a == b => {
                    i += 1;
                    j += 1;
                    let product = self.multiply(bv, factor)?;
                    (*a, self.add(av, &product)?)
                }
                (Some((a, av)), other) if other.is_none_or(|(b, _)| a < b) => {
                    i += 1;
                    (*a, self.clone_value(av)?)
                }
                (_, Some((b, bv))) => {
                    j += 1;
                    (*b, self.multiply(bv, factor)?)
                }
                _ => unreachable!("one sparse row has a remaining entry"),
            };
            if !value.is_zero() {
                if result.len() == capacity {
                    return Err(SparseRrefError::Limit("row replay fill"));
                }
                result.push((column, value));
            }
        }
        Ok(result)
    }
    fn digest(&mut self, matrix: &RationalSparseMatrix) -> Result<[u8; 32]> {
        let mut hash =
            blake3::Hasher::new_derive_key("frankensympy sparse rational matrix schema 1");
        for size in [matrix.rows, matrix.columns, matrix.entries.len()] {
            hash.update(&(size as u64).to_le_bytes());
        }
        for entry in &matrix.entries {
            self.tick(1)?;
            hash.update(&(entry.row as u64).to_le_bytes());
            hash.update(&(entry.column as u64).to_le_bytes());
            for integer in [entry.value.numer(), entry.value.denom()] {
                self.tick(integer.limb_count())?;
                self.storage(integer.bits().div_ceil(8))?;
                let bytes = integer.to_bytes_le();
                hash.update(&[u8::from(integer.is_negative())]);
                hash.update(&(bytes.len() as u64).to_le_bytes());
                hash.update(&bytes);
            }
        }
        Ok(*hash.finalize().as_bytes())
    }
}

/// Verify the complete exact relation `source ~ reduced` and the reduced
/// matrix's canonical pivot/rank claim. No floating-point decisions or
/// optimizing generator code participate in acceptance.
pub fn verify_sparse_rref<M: BudgetMeter>(
    source: &RationalSparseMatrix,
    witness: &SparseRrefWitness,
    meter: &mut M,
) -> Result<SparseRrefReceipt> {
    let mut work = Reference {
        meter,
        steps: 0,
        bytes: 0,
    };
    verify_sparse_rref_reference(source, witness, &mut work)
}

pub(super) fn verify_sparse_rref_reference<M: BudgetMeter>(
    source: &RationalSparseMatrix,
    witness: &SparseRrefWitness,
    work: &mut Reference<'_, M>,
) -> Result<SparseRrefReceipt> {
    work.tick(1)?;
    if witness.schema_version != SPARSE_RREF_SCHEMA_VERSION {
        return Err(SparseRrefError::Invalid("unknown witness schema"));
    }
    work.check_matrix(source)?;
    work.check_matrix(&witness.reduced)?;
    if (source.rows, source.columns) != (witness.reduced.rows, witness.reduced.columns)
        || witness.pivots.len() > source.rows.min(source.columns)
    {
        return Err(SparseRrefError::Invalid("witness shape"));
    }
    if witness.operations.len() > MAX_SPARSE_RREF_OPERATIONS {
        return Err(SparseRrefError::Limit("row-operation count"));
    }
    for operation in &witness.operations {
        work.tick(1)?;
        match operation {
            LinearRowOperation::Swap { left, right } => {
                if *left >= source.rows || *right >= source.rows {
                    return Err(SparseRrefError::Invalid("swap row index"));
                }
            }
            LinearRowOperation::Scale { row, factor } => {
                check_scalar(factor)?;
                if *row >= source.rows || factor.is_zero() {
                    return Err(SparseRrefError::Invalid("noninvertible row scale"));
                }
            }
            LinearRowOperation::Add {
                target,
                source: from,
                factor,
            } => {
                check_scalar(factor)?;
                if *target >= source.rows || *from >= source.rows || target == from {
                    return Err(SparseRrefError::Invalid("noninvertible row addition"));
                }
            }
        }
    }
    let mut pivot_rows = work.vector(source.columns)?;
    pivot_rows.resize(source.columns, None);
    let mut previous = None;
    for (row, &column) in witness.pivots.iter().enumerate() {
        work.tick(1)?;
        if column >= source.columns || previous.is_some_and(|last| last >= column) {
            return Err(SparseRrefError::Invalid("pivot order or coordinate"));
        }
        pivot_rows[column] = Some(row);
        previous = Some(column);
    }
    let mut rows = work.rows(source)?;
    let mut stored = source.entries.len();
    for operation in &witness.operations {
        work.tick(1)?;
        match operation {
            LinearRowOperation::Swap { left, right } => rows.swap(*left, *right),
            LinearRowOperation::Scale { row, factor } => {
                for (_, value) in &mut rows[*row] {
                    *value = work.multiply(value, factor)?;
                }
            }
            LinearRowOperation::Add {
                target,
                source: from,
                factor,
            } => {
                if !factor.is_zero() {
                    let unaffected = stored - rows[*target].len();
                    let row = work.add_row(
                        &rows[*target],
                        &rows[*from],
                        factor,
                        MAX_SPARSE_RREF_ENTRIES - unaffected,
                    )?;
                    stored = unaffected + row.len();
                    rows[*target] = row;
                }
            }
        }
    }
    // Bind every resulting cell to the supplied artifact and independently
    // check every RREF condition, including all zero rows and pivot columns.
    let mut expected = witness.reduced.entries.iter();
    for (row, entries) in rows.iter().enumerate() {
        work.tick(1)?;
        match witness.pivots.get(row) {
            Some(&column) => match entries.first() {
                Some((first, value)) if *first == column && value.is_one() => {}
                _ => return Err(SparseRrefError::Invalid("missing or nonunit leading pivot")),
            },
            None if entries.is_empty() => {}
            None => return Err(SparseRrefError::Invalid("nonzero row after claimed rank")),
        }
        for (column, value) in entries {
            work.tick(value.numer().limb_count() + value.denom().limb_count())?;
            let entry = expected
                .next()
                .ok_or(SparseRrefError::Invalid("truncated reduced matrix"))?;
            if (entry.row, entry.column) != (row, *column) || &entry.value != value {
                return Err(SparseRrefError::Invalid(
                    "row replay disagrees with reduced matrix",
                ));
            }
            if pivot_rows[*column].is_some_and(|pivot_row| pivot_row != row) {
                return Err(SparseRrefError::Invalid("nonzero off-pivot column entry"));
            }
        }
    }
    if expected.next().is_some() {
        return Err(SparseRrefError::Invalid("extra reduced matrix entries"));
    }
    let source_digest = work.digest(source)?;
    let reduced_digest = work.digest(&witness.reduced)?;
    let mut pivots = work.vector(witness.pivots.len())?;
    pivots.extend_from_slice(&witness.pivots);
    work.tick(1)?;
    Ok(SparseRrefReceipt {
        rows: source.rows,
        columns: source.columns,
        pivots,
        source_digest,
        reduced_digest,
    })
}

fn affine_augmented_source<M: BudgetMeter>(
    coefficients: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    work: &mut Reference<'_, M>,
) -> Result<RationalSparseMatrix> {
    work.check_matrix(coefficients)?;
    work.check_matrix(rhs)?;
    if coefficients.rows != rhs.rows {
        return Err(SparseRrefError::Invalid("affine source row mismatch"));
    }
    let columns = coefficients
        .columns
        .checked_add(rhs.columns)
        .filter(|&columns| columns <= MAX_SPARSE_RREF_DIMENSION)
        .ok_or(SparseRrefError::Limit("affine augmented dimensions"))?;
    let count = coefficients
        .entries
        .len()
        .checked_add(rhs.entries.len())
        .filter(|&count| count <= MAX_SPARSE_RREF_ENTRIES)
        .ok_or(SparseRrefError::Limit("affine augmented stored entries"))?;
    let mut entries = work.vector(count)?;
    let (mut left, mut right) = (0, 0);
    while left < coefficients.entries.len() || right < rhs.entries.len() {
        work.tick(1)?;
        let use_left = right == rhs.entries.len()
            || (left < coefficients.entries.len()
                && coefficients.entries[left].row <= rhs.entries[right].row);
        let (entry, offset) = if use_left {
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

/// Multiply original sparse rows by independently supplied output rows. No
/// reduced RHS or elimination transcript participates in the expected value.
fn check_affine_product<M: BudgetMeter>(
    coefficients: &[Row],
    output: &[Row],
    expected: Option<&[Row]>,
    columns: usize,
    work: &mut Reference<'_, M>,
) -> Result<()> {
    for (row, coefficients) in coefficients.iter().enumerate() {
        work.tick(1)?;
        let mut product = work.vector(0)?;
        for (variable, coefficient) in coefficients {
            product = work.add_row(&product, &output[*variable], coefficient, columns)?;
        }
        let expected = expected.map_or(&[][..], |rows| rows[row].as_slice());
        if product.len() != expected.len() {
            return Err(SparseRrefError::Invalid("affine original-source residual"));
        }
        for ((column, value), (expected_column, expected_value)) in product.iter().zip(expected) {
            work.tick(
                value.numer().limb_count()
                    + value.denom().limb_count()
                    + expected_value.numer().limb_count()
                    + expected_value.denom().limb_count(),
            )?;
            if column != expected_column || value != expected_value {
                return Err(SparseRrefError::Invalid("affine original-source residual"));
            }
        }
    }
    work.tick(1)?;
    Ok(())
}

fn check_affine_outputs<M: BudgetMeter>(
    coefficients: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    particular: &RationalSparseMatrix,
    kernel: &RationalSparseMatrix,
    free_columns: &[usize],
    work: &mut Reference<'_, M>,
) -> Result<()> {
    work.check_matrix(particular)?;
    work.check_matrix(kernel)?;
    if particular.rows != coefficients.columns
        || particular.columns != rhs.columns
        || kernel.rows != coefficients.columns
        || kernel.columns != free_columns.len()
    {
        return Err(SparseRrefError::Invalid("affine output shape"));
    }
    if particular
        .entries
        .len()
        .saturating_add(kernel.entries.len())
        > MAX_SPARSE_RREF_ENTRIES
    {
        return Err(SparseRrefError::Limit("aggregate affine output entries"));
    }
    let mut free_indices = work.vector(coefficients.columns)?;
    free_indices.resize(coefficients.columns, None);
    for (index, &column) in free_columns.iter().enumerate() {
        work.tick(1)?;
        free_indices[column] = Some(index);
    }
    for entry in &particular.entries {
        work.tick(1)?;
        if free_indices[entry.row].is_some() {
            return Err(SparseRrefError::Invalid(
                "particular free coordinate is nonzero",
            ));
        }
    }
    let mut identities = work.vector(free_columns.len())?;
    identities.resize(free_columns.len(), false);
    for entry in &kernel.entries {
        work.tick(1)?;
        if let Some(index) = free_indices[entry.row] {
            if entry.column != index || !entry.value.is_one() {
                return Err(SparseRrefError::Invalid("kernel free-coordinate identity"));
            }
            identities[index] = true;
        }
    }
    for present in identities {
        work.tick(1)?;
        if !present {
            return Err(SparseRrefError::Invalid("missing kernel free coordinate"));
        }
    }
    let coefficient_rows = work.rows(coefficients)?;
    let rhs_rows = work.rows(rhs)?;
    let particular_rows = work.rows(particular)?;
    let kernel_rows = work.rows(kernel)?;
    check_affine_product(
        &coefficient_rows,
        &particular_rows,
        Some(&rhs_rows),
        rhs.columns,
        work,
    )?;
    check_affine_product(
        &coefficient_rows,
        &kernel_rows,
        None,
        free_columns.len(),
        work,
    )
}

/// Independently verify complete exact sparse affine solutions to `A * X = B`.
/// The original sources, augmented replay, complete kernel, original residuals,
/// commitments and publication all share one cumulative verification ledger.
pub fn verify_sparse_affine<M: BudgetMeter>(
    coefficients: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    witness: &SparseAffineWitness,
    meter: &mut M,
) -> Result<SparseAffineReceipt> {
    let mut work = Reference {
        meter,
        steps: 0,
        bytes: 0,
    };
    work.tick(1)?;
    if witness.schema_version != SPARSE_AFFINE_SCHEMA_VERSION {
        return Err(SparseRrefError::Invalid("unknown affine witness schema"));
    }
    let augmented = affine_augmented_source(coefficients, rhs, &mut work)?;
    let reduction = verify_sparse_rref_reference(&augmented, &witness.reduction, &mut work)?;
    let rank = reduction
        .pivots()
        .partition_point(|&column| column < coefficients.columns);
    let mut free_columns = work.vector(coefficients.columns - rank)?;
    let mut next_pivot = 0;
    for column in 0..coefficients.columns {
        work.tick(1)?;
        if reduction.pivots().get(next_pivot) == Some(&column) {
            next_pivot += 1;
        } else {
            free_columns.push(column);
        }
    }
    let (contradiction_row, particular_digest, kernel_digest) = match &witness.outcome {
        SparseAffineOutcome::Inconsistent { row } => {
            work.tick(1)?;
            if reduction
                .pivots()
                .get(*row)
                .is_none_or(|&column| column < coefficients.columns)
            {
                return Err(SparseRrefError::Invalid(
                    "affine contradiction row is not contradictory",
                ));
            }
            (Some(*row), None, None)
        }
        SparseAffineOutcome::Solvable {
            particular,
            kernel,
            free_columns: claimed_free,
        } => {
            work.tick(1)?;
            if rank != reduction.rank() {
                return Err(SparseRrefError::Invalid(
                    "affine solution omits an inconsistent RHS",
                ));
            }
            if claimed_free.len() != free_columns.len() {
                return Err(SparseRrefError::Invalid("affine free-column count"));
            }
            work.tick(free_columns.len() as u64)?;
            if claimed_free != &free_columns {
                return Err(SparseRrefError::Invalid("affine free-column complement"));
            }
            check_affine_outputs(
                coefficients,
                rhs,
                particular,
                kernel,
                &free_columns,
                &mut work,
            )?;
            (
                None,
                Some(work.digest(particular)?),
                Some(work.digest(kernel)?),
            )
        }
    };
    let source_digest = work.digest(coefficients)?;
    let rhs_digest = work.digest(rhs)?;
    work.tick(1)?;
    Ok(SparseAffineReceipt {
        rows: coefficients.rows,
        columns: coefficients.columns,
        rhs_columns: rhs.columns,
        rank,
        augmented_rank: reduction.rank(),
        free_columns,
        contradiction_row,
        source_digest,
        rhs_digest,
        particular_digest,
        kernel_digest,
    })
}
