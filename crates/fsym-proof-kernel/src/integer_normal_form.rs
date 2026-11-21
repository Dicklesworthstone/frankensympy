//! Independent Smith normal form certificates over the integers.
//!
//! The precise claim is `D = U A V`, where U and V are unimodular integer
//! matrices and D has positive nonzero diagonal entries dividing their
//! successors, followed by zeros. Inverse witnesses are checked in both
//! directions over Z. A rational inverse or a product identity alone cannot
//! establish unimodularity or a complete integer-module normal form.
//!
//! Verification uses bounded sparse reference dot products, without pivot
//! selection, Euclidean elimination, or a dependency on a matrix generator.
//! All admission, arithmetic, copying, hashing and publication shares the
//! caller's meter and a cumulative 20M-step/256MiB reference envelope. The
//! caller supplies protected verifier resources when its policy requires them.

use crate::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, RationalSparseMatrix, Reference, SparseRationalEntry,
    SparseRrefError,
};
use fsym_budget::BudgetMeter;
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use thiserror::Error;

pub const INTEGER_SMITH_SCHEMA_VERSION: u32 = 1;
pub const MAX_INTEGER_SMITH_DIMENSION: usize = 64;
pub const MAX_INTEGER_SMITH_COEFFICIENT_BITS: u64 = MAX_SPARSE_RREF_COEFFICIENT_BITS;
const MAX_ENTRIES: usize = MAX_INTEGER_SMITH_DIMENSION * MAX_INTEGER_SMITH_DIMENSION;

/// An untrusted proposal for an integer Smith normal form. All five matrices
/// have canonical sparse storage and integral coefficients. The receiver
/// supplies A separately, rather than trusting a source recorded in the proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerSmithWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub diagonal: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub left: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub right: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub left_inverse: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub right_inverse: RationalSparseMatrix,
}

/// Accepted complete integer-module normal form. No public constructor or
/// deserializer can manufacture this receipt. Digests bind the exact original
/// source and witness; retain those values for independent replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerSmithReceipt {
    rows: usize,
    columns: usize,
    invariant_factors: Vec<BigInt>,
    source_digest: [u8; 32],
    diagonal_digest: [u8; 32],
    left_digest: [u8; 32],
    right_digest: [u8; 32],
    left_inverse_digest: [u8; 32],
    right_inverse_digest: [u8; 32],
}

impl IntegerSmithReceipt {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rank(&self) -> usize {
        self.invariant_factors.len()
    }
    pub fn invariant_factors(&self) -> &[BigInt] {
        &self.invariant_factors
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn diagonal_digest(&self) -> [u8; 32] {
        self.diagonal_digest
    }
    pub fn left_digest(&self) -> [u8; 32] {
        self.left_digest
    }
    pub fn right_digest(&self) -> [u8; 32] {
        self.right_digest
    }
    pub fn left_inverse_digest(&self) -> [u8; 32] {
        self.left_inverse_digest
    }
    pub fn right_inverse_digest(&self) -> [u8; 32] {
        self.right_inverse_digest
    }
}

#[derive(Debug, Error)]
pub enum IntegerSmithError {
    #[error("integer Smith normal form certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

impl From<RationalArithmeticError> for IntegerSmithError {
    fn from(error: RationalArithmeticError) -> Self {
        Self::Reference(error.into())
    }
}

type Result<T> = std::result::Result<T, IntegerSmithError>;
type Row = Vec<(usize, BigRational)>;

fn bounded_dimension<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<usize, D::Error> {
    let value = usize::deserialize(d)?;
    if value > MAX_INTEGER_SMITH_DIMENSION {
        return Err(serde::de::Error::custom("integer Smith matrix dimension"));
    }
    Ok(value)
}

fn bounded_entries<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<SparseRationalEntry>, D::Error> {
    struct Entries;
    impl<'de> Visitor<'de> for Entries {
        type Value = Vec<SparseRationalEntry>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "at most {MAX_ENTRIES} integer matrix entries")
        }

        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            if sequence.size_hint().is_some_and(|n| n > MAX_ENTRIES) {
                return Err(serde::de::Error::custom("integer Smith entry count"));
            }
            let mut entries = Vec::new();
            loop {
                if entries.len() == MAX_ENTRIES {
                    if sequence.next_element::<IgnoredAny>()?.is_some() {
                        return Err(serde::de::Error::custom("integer Smith entry count"));
                    }
                    return Ok(entries);
                }
                let Some(entry) = sequence.next_element::<SparseRationalEntry>()? else {
                    return Ok(entries);
                };
                // SparseRationalEntry bounds each integer before rational
                // normalization. Refuse a noninteger before retaining it or
                // decoding the next potentially large witness coefficient.
                if !entry.value.is_integer() {
                    return Err(serde::de::Error::custom("noninteger Smith coefficient"));
                }
                entries.push(entry);
            }
        }
    }
    d.deserialize_seq(Entries)
}

pub(super) fn bounded_integer_matrix<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<RationalSparseMatrix, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Matrix {
        #[serde(deserialize_with = "bounded_dimension")]
        rows: usize,
        #[serde(deserialize_with = "bounded_dimension")]
        columns: usize,
        #[serde(deserialize_with = "bounded_entries")]
        entries: Vec<SparseRationalEntry>,
    }
    let matrix = Matrix::deserialize(d)?;
    Ok(RationalSparseMatrix {
        rows: matrix.rows,
        columns: matrix.columns,
        entries: matrix.entries,
    })
}

pub(super) fn admit_integer_matrix(
    matrix: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    if matrix.rows > MAX_INTEGER_SMITH_DIMENSION
        || matrix.columns > MAX_INTEGER_SMITH_DIMENSION
        || matrix.entries.len() > MAX_ENTRIES
    {
        return Err(SparseRrefError::Limit("integer Smith matrix envelope").into());
    }
    work.check_matrix(matrix)?;
    for entry in &matrix.entries {
        work.tick(1)?;
        if !entry.value.is_integer() {
            return Err(IntegerSmithError::Invalid("noninteger coefficient"));
        }
    }
    Ok(())
}

fn transpose(
    rows: &[Row],
    columns: usize,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Vec<Row>> {
    let mut result = work.vector(columns)?;
    for _ in 0..columns {
        result.push(work.vector(rows.len())?);
    }
    for (row, entries) in rows.iter().enumerate() {
        for (column, value) in entries {
            work.tick(1)?;
            result[*column].push((row, work.clone_value(value)?));
        }
    }
    Ok(result)
}

// Straight sparse dot products are intentionally independent of the
// generator's row/column elimination and transformation-update machinery.
fn dot(left: &Row, right: &Row, work: &mut Reference<'_, impl BudgetMeter>) -> Result<BigRational> {
    work.storage(std::mem::size_of::<BigRational>() as u64)?;
    let mut sum = BigRational::zero();
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        work.tick(1)?;
        match left[i].0.cmp(&right[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let term = work.multiply(&left[i].1, &right[j].1)?;
                sum = work.add(&sum, &term)?;
                i += 1;
                j += 1;
            }
        }
    }
    Ok(sum)
}

pub(super) fn check_inverse_product(
    left: &[Row],
    right: &[Row],
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    let columns = transpose(right, left.len(), work)?;
    for (row, entries) in left.iter().enumerate() {
        for (column, right) in columns.iter().enumerate() {
            let value = dot(entries, right, work)?;
            work.tick(value.numer().limb_count() + value.denom().limb_count())?;
            if !(if row == column {
                value.is_one()
            } else {
                value.is_zero()
            }) {
                return Err(IntegerSmithError::Invalid(
                    "integer transformation inverse identity does not hold",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn multiply(
    left: &[Row],
    right: &[Row],
    columns: usize,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Vec<Row>> {
    let right_columns = transpose(right, columns, work)?;
    let mut result = work.vector(left.len())?;
    for row in left {
        let mut values = work.vector(columns)?;
        for (column, right) in right_columns.iter().enumerate() {
            let value = dot(row, right, work)?;
            if !value.is_zero() {
                values.push((column, value));
            }
        }
        result.push(values);
    }
    Ok(result)
}

fn check_product(
    left: &[Row],
    right: &[Row],
    expected: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    let columns = transpose(right, expected.columns, work)?;
    let mut next = expected.entries.iter().peekable();
    for (row, entries) in left.iter().enumerate() {
        for (column, right) in columns.iter().enumerate() {
            let value = dot(entries, right, work)?;
            let agrees = if next
                .peek()
                .is_some_and(|entry| (entry.row, entry.column) == (row, column))
            {
                let entry = next.next().expect("entry inspected");
                work.tick(
                    value.numer().limb_count()
                        + value.denom().limb_count()
                        + entry.value.numer().limb_count(),
                )?;
                value == entry.value
            } else {
                value.is_zero()
            };
            if !agrees {
                return Err(IntegerSmithError::Invalid("U A V differs from D"));
            }
        }
    }
    if next.next().is_some() {
        return Err(IntegerSmithError::Invalid("unmatched diagonal entries"));
    }
    Ok(())
}

/// Verify a complete Smith normal form of the separately supplied integer A.
/// Rectangular, rank-deficient and zero-area matrices are supported. Unknown
/// schemas, noninteger transformations, false inverse/reconstruction identities,
/// noncanonical invariant factors and interrupted work yield no receipt.
pub fn verify_integer_smith(
    source: &RationalSparseMatrix,
    witness: &IntegerSmithWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<IntegerSmithReceipt> {
    let mut work = Reference::new(verifier);
    verify_integer_smith_with_work(source, witness, &mut work)
}

// Larger certificates share this exact checker and its cumulative reference
// envelope, rather than obtaining fresh internal limits for each subproof.
pub(super) fn verify_integer_smith_with_work(
    source: &RationalSparseMatrix,
    witness: &IntegerSmithWitness,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<IntegerSmithReceipt> {
    work.tick(1)?;
    if witness.schema_version != INTEGER_SMITH_SCHEMA_VERSION {
        return Err(IntegerSmithError::Invalid("unknown witness schema"));
    }
    for matrix in [
        source,
        &witness.diagonal,
        &witness.left,
        &witness.right,
        &witness.left_inverse,
        &witness.right_inverse,
    ] {
        admit_integer_matrix(matrix, work)?;
    }
    let (m, n) = (source.rows, source.columns);
    if (witness.diagonal.rows, witness.diagonal.columns) != (m, n)
        || [&witness.left, &witness.left_inverse]
            .iter()
            .any(|matrix| (matrix.rows, matrix.columns) != (m, m))
        || [&witness.right, &witness.right_inverse]
            .iter()
            .any(|matrix| (matrix.rows, matrix.columns) != (n, n))
    {
        return Err(IntegerSmithError::Invalid("Smith transformation shape"));
    }

    let mut previous: Option<&BigRational> = None;
    for (index, entry) in witness.diagonal.entries.iter().enumerate() {
        work.tick(1)?;
        if (entry.row, entry.column) != (index, index) || !entry.value.numer().is_positive() {
            return Err(IntegerSmithError::Invalid(
                "D requires positive diagonal entries followed by zeros",
            ));
        }
        if let Some(previous) = previous {
            work.tick(1)?;
            if !entry.value.metered_rem(previous, work)?.is_zero() {
                return Err(IntegerSmithError::Invalid(
                    "Smith invariant factors do not divide their successors",
                ));
            }
        }
        previous = Some(&entry.value);
    }

    let left = work.rows(&witness.left)?;
    let right = work.rows(&witness.right)?;
    let left_inverse = work.rows(&witness.left_inverse)?;
    let right_inverse = work.rows(&witness.right_inverse)?;
    check_inverse_product(&left, &left_inverse, work)?;
    check_inverse_product(&left_inverse, &left, work)?;
    check_inverse_product(&right, &right_inverse, work)?;
    check_inverse_product(&right_inverse, &right, work)?;

    let original = work.rows(source)?;
    let left_original = multiply(&left, &original, n, work)?;
    check_product(&left_original, &right, &witness.diagonal, work)?;

    let source_digest = work.digest(source)?;
    let diagonal_digest = work.digest(&witness.diagonal)?;
    let left_digest = work.digest(&witness.left)?;
    let right_digest = work.digest(&witness.right)?;
    let left_inverse_digest = work.digest(&witness.left_inverse)?;
    let right_inverse_digest = work.digest(&witness.right_inverse)?;
    let mut invariant_factors = work.vector(witness.diagonal.entries.len())?;
    for entry in &witness.diagonal.entries {
        let integer = entry.value.numer();
        work.tick(integer.limb_count())?;
        work.storage(integer.limb_count().saturating_mul(8))?;
        invariant_factors.push(integer.clone());
    }
    work.tick(1)?;
    Ok(IntegerSmithReceipt {
        rows: m,
        columns: n,
        invariant_factors,
        source_digest,
        diagonal_digest,
        left_digest,
        right_digest,
        left_inverse_digest,
        right_inverse_digest,
    })
}
