//! Independent certificates for exact LLL reduction of an integer row lattice.
//!
//! For the separately supplied integer source A and exact rational delta, the
//! claim is B = T A with an integral, two-sided inverse for T. Reference
//! Gram--Schmidt arithmetic then establishes |mu[i,j]| <= 1/2 and every
//! Lovasz inequality for that same delta. This proves equality of the complete
//! integer row lattices and delta-LLL reducedness, not a canonical basis or
//! shortest vectors. The checker never runs the generator's reduction search.
//!
//! The regular envelope is a full-row-rank m-by-n basis with m <= 16 and
//! m <= n <= 64. Empty bases and the compatible one-zero-row case are accepted
//! with rank zero; no full-row-rank claim is made for that exceptional row.
//! Admission, rational arithmetic, copies, hashing and publication share one
//! cumulative 20M-step/256MiB reference envelope and the caller's meter.

use crate::integer_normal_form::{
    IntegerSmithError, MAX_INTEGER_SMITH_COEFFICIENT_BITS, admit_integer_matrix,
    check_inverse_product, multiply,
};
use crate::sparse_rref::{
    RationalSparseMatrix, Reference, SparseRationalEntry, SparseRrefError, check_scalar,
};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use fsym_core::arith::RationalArithmeticError;
use num_traits::{One, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::cmp::Ordering;
use std::fmt;
use thiserror::Error;

pub const INTEGER_LLL_SCHEMA_VERSION: u32 = 1;
pub const MAX_INTEGER_LLL_ROWS: usize = 16;
pub const MAX_INTEGER_LLL_COLUMNS: usize = 64;
pub const MAX_INTEGER_LLL_COEFFICIENT_BITS: u64 = MAX_INTEGER_SMITH_COEFFICIENT_BITS;
const MAX_ENTRIES: usize = MAX_INTEGER_LLL_ROWS * MAX_INTEGER_LLL_COLUMNS;

/// Untrusted portable row-lattice witness. Both the original source and delta
/// are independent verifier inputs, rather than claims chosen by the witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerLllWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_matrix")]
    pub basis: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_matrix")]
    pub transform: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_matrix")]
    pub transform_inverse: RationalSparseMatrix,
}

/// Accepted complete row-lattice equivalence and exact delta-LLL reducedness.
/// There is no unchecked constructor or deserializer for an accepted receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerLllReceipt {
    rows: usize,
    columns: usize,
    rank: usize,
    delta: BigRational,
    source_digest: [u8; 32],
    basis_digest: [u8; 32],
    transform_digest: [u8; 32],
    transform_inverse_digest: [u8; 32],
}

impl IntegerLllReceipt {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn delta(&self) -> &BigRational {
        &self.delta
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn basis_digest(&self) -> [u8; 32] {
        self.basis_digest
    }
    pub fn transform_digest(&self) -> [u8; 32] {
        self.transform_digest
    }
    pub fn transform_inverse_digest(&self) -> [u8; 32] {
        self.transform_inverse_digest
    }
}

#[derive(Debug, Error)]
pub enum IntegerLllError {
    #[error("integer LLL certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

impl From<IntegerSmithError> for IntegerLllError {
    fn from(error: IntegerSmithError) -> Self {
        match error {
            IntegerSmithError::Invalid(reason) => Self::Invalid(reason),
            IntegerSmithError::Reference(error) => Self::Reference(error),
        }
    }
}

impl From<RationalArithmeticError> for IntegerLllError {
    fn from(error: RationalArithmeticError) -> Self {
        Self::Reference(error.into())
    }
}

type Result<T> = std::result::Result<T, IntegerLllError>;
type Row = Vec<(usize, BigRational)>;

fn bounded_dimension<'de, D: Deserializer<'de>, const MAX: usize>(
    d: D,
) -> std::result::Result<usize, D::Error> {
    let value = usize::deserialize(d)?;
    if value > MAX {
        return Err(serde::de::Error::custom("integer LLL matrix dimension"));
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
            write!(f, "at most {MAX_ENTRIES} bounded integer LLL entries")
        }

        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            if sequence.size_hint().is_some_and(|n| n > MAX_ENTRIES) {
                return Err(serde::de::Error::custom("integer LLL entry count"));
            }
            let mut entries = Vec::new();
            loop {
                if entries.len() == MAX_ENTRIES {
                    if sequence.next_element::<IgnoredAny>()?.is_some() {
                        return Err(serde::de::Error::custom("integer LLL entry count"));
                    }
                    return Ok(entries);
                }
                let Some(entry) = sequence.next_element::<SparseRationalEntry>()? else {
                    return Ok(entries);
                };
                // The nested entry decoder bounds both integers before any
                // rational normalization. Reject fractions before retaining.
                if !entry.value.is_integer() {
                    return Err(serde::de::Error::custom("noninteger LLL coefficient"));
                }
                entries.push(entry);
            }
        }
    }
    d.deserialize_seq(Entries)
}

fn bounded_matrix<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<RationalSparseMatrix, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Matrix {
        #[serde(deserialize_with = "bounded_dimension::<_, MAX_INTEGER_LLL_ROWS>")]
        rows: usize,
        #[serde(deserialize_with = "bounded_dimension::<_, MAX_INTEGER_LLL_COLUMNS>")]
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

fn admit(
    matrix: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    if matrix.rows > MAX_INTEGER_LLL_ROWS
        || matrix.columns > MAX_INTEGER_LLL_COLUMNS
        || matrix.entries.len() > MAX_ENTRIES
    {
        return Err(SparseRrefError::Limit("integer LLL matrix envelope").into());
    }
    admit_integer_matrix(matrix, work)?;
    Ok(())
}

fn subtract(
    left: &BigRational,
    right: &BigRational,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let negative = -work.clone_value(right)?;
    Ok(work.add(left, &negative)?)
}

fn divide(
    numerator: &BigRational,
    denominator: &BigRational,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    if numerator
        .numer()
        .bits()
        .saturating_add(denominator.denom().bits())
        > MAX_INTEGER_LLL_COEFFICIENT_BITS
        || numerator
            .denom()
            .bits()
            .saturating_add(denominator.numer().bits())
            > MAX_INTEGER_LLL_COEFFICIENT_BITS
    {
        return Err(SparseRrefError::Limit("LLL reciprocal intermediate height").into());
    }
    work.tick(1)?;
    let value = numerator.metered_div(denominator, work)?;
    check_scalar(&value)?;
    Ok(value)
}

fn dot(left: &Row, right: &Row, work: &mut Reference<'_, impl BudgetMeter>) -> Result<BigRational> {
    work.storage(std::mem::size_of::<BigRational>() as u64)?;
    let mut sum = BigRational::zero();
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        work.tick(1)?;
        match left[i].0.cmp(&right[j].0) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                let product = work.multiply(&left[i].1, &right[j].1)?;
                sum = work.add(&sum, &product)?;
                i += 1;
                j += 1;
            }
        }
    }
    Ok(sum)
}

/// Compute Gram--Schmidt coefficients from the exact Gram entries by their
/// defining triangular recurrence. No row-reduction trace, rounding rule or
/// Gram--Schmidt data supplied by the generator participates in acceptance.
fn check_reduced(
    rows: &[Row],
    delta: &BigRational,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<usize> {
    work.storage(128)?;
    let half = BigRational::new(1.into(), 2.into());
    let negative_half = -work.clone_value(&half)?;
    let mut coefficients: Vec<Vec<BigRational>> = work.vector(rows.len())?;
    let mut norms: Vec<BigRational> = work.vector(rows.len())?;
    for (i, row) in rows.iter().enumerate() {
        work.tick(1)?;
        let mut current: Vec<BigRational> = work.vector(i)?;
        let mut norm = dot(row, row, work)?;
        for j in 0..i {
            let mut projection = dot(row, &rows[j], work)?;
            for k in 0..j {
                let product = work.multiply(&current[k], &coefficients[j][k])?;
                let product = work.multiply(&product, &norms[k])?;
                projection = subtract(&projection, &product, work)?;
            }
            // Earlier norms have already been proven strictly positive.
            let mu = divide(&projection, &norms[j], work)?;
            if mu.metered_cmp(&half, work)? == Ordering::Greater
                || mu.metered_cmp(&negative_half, work)? == Ordering::Less
            {
                return Err(IntegerLllError::Invalid("basis is not size reduced"));
            }
            let square = work.multiply(&mu, &mu)?;
            let removed = work.multiply(&square, &norms[j])?;
            norm = subtract(&norm, &removed, work)?;
            current.push(mu);
        }
        if !norm.numer().is_positive() {
            if rows.len() == 1 && norm.is_zero() {
                return Ok(0);
            }
            return Err(IntegerLllError::Invalid("basis rows are linearly dependent"));
        }
        if i > 0 {
            let previous_mu = &current[i - 1];
            let square = work.multiply(previous_mu, previous_mu)?;
            let factor = subtract(delta, &square, work)?;
            let lower_bound = work.multiply(&factor, &norms[i - 1])?;
            if norm.metered_cmp(&lower_bound, work)? == Ordering::Less {
                return Err(IntegerLllError::Invalid("basis violates Lovasz inequality"));
            }
        }
        coefficients.push(current);
        norms.push(norm);
    }
    Ok(rows.len())
}

/// Independently check complete lattice equality and reducedness for the exact
/// caller-supplied delta in (1/4, 1). Cancellation or any admission, resource,
/// inverse, source-product or reducedness failure returns no accepted receipt.
pub fn verify_integer_lll(
    source: &RationalSparseMatrix,
    delta: &BigRational,
    witness: &IntegerLllWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<IntegerLllReceipt> {
    let mut work = Reference::new(verifier);
    work.tick(1)?;
    if witness.schema_version != INTEGER_LLL_SCHEMA_VERSION {
        return Err(IntegerLllError::Invalid("unknown witness schema"));
    }
    check_scalar(delta)?;
    work.storage(128)?;
    let quarter = BigRational::new(1.into(), 4.into());
    let one = BigRational::one();
    if delta.metered_cmp(&quarter, &mut work)? != Ordering::Greater
        || delta.metered_cmp(&one, &mut work)? != Ordering::Less
    {
        return Err(IntegerLllError::Invalid(
            "delta must lie strictly between 1/4 and 1",
        ));
    }
    for matrix in [
        source,
        &witness.basis,
        &witness.transform,
        &witness.transform_inverse,
    ] {
        admit(matrix, &mut work)?;
    }
    let (m, n) = (source.rows, source.columns);
    if m > n
        || (witness.basis.rows, witness.basis.columns) != (m, n)
        || [&witness.transform, &witness.transform_inverse]
            .iter()
            .any(|matrix| (matrix.rows, matrix.columns) != (m, m))
    {
        return Err(IntegerLllError::Invalid("LLL transformation shape"));
    }
    let transform = work.rows(&witness.transform)?;
    let transform_inverse = work.rows(&witness.transform_inverse)?;
    check_inverse_product(&transform, &transform_inverse, &mut work)?;
    check_inverse_product(&transform_inverse, &transform, &mut work)?;

    let original = work.rows(source)?;
    let product = multiply(&transform, &original, n, &mut work)?;
    let mut expected = witness.basis.entries.iter();
    for (row, entries) in product.iter().enumerate() {
        for (column, value) in entries {
            work.tick(1)?;
            let Some(entry) = expected.next() else {
                return Err(IntegerLllError::Invalid(
                    "T A has an omitted lattice direction",
                ));
            };
            work.tick(value.numer().limb_count() + entry.value.numer().limb_count())?;
            if (row, *column) != (entry.row, entry.column) || value != &entry.value {
                return Err(IntegerLllError::Invalid("T A differs from the reduced basis"));
            }
        }
    }
    if expected.next().is_some() {
        return Err(IntegerLllError::Invalid("unmatched reduced basis entries"));
    }

    let basis = work.rows(&witness.basis)?;
    let rank = check_reduced(&basis, delta, &mut work)?;
    let source_digest = work.digest(source)?;
    let basis_digest = work.digest(&witness.basis)?;
    let transform_digest = work.digest(&witness.transform)?;
    let transform_inverse_digest = work.digest(&witness.transform_inverse)?;
    let delta = work.clone_value(delta)?;
    work.tick(1)?;
    Ok(IntegerLllReceipt {
        rows: m,
        columns: n,
        rank,
        delta,
        source_digest,
        basis_digest,
        transform_digest,
        transform_inverse_digest,
    })
}
