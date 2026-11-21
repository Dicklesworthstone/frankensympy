//! Independent certificates for the column Hermite normal form over Z.
//!
//! For a separately supplied m-by-n integer source A, the exact claim is
//! `A V = [0 | H]`, where V has an integer inverse and H is the canonical
//! m-by-r column Hermite form. Pivots are the last nonzero entry of each
//! column, have strictly increasing row indices and are positive. Entries
//! to the right of each pivot lie in [0, pivot). No zero column is retained.
//!
//! These conditions establish a complete basis of the original column
//! lattice, not merely a rationally equivalent span or a proper sublattice.
//! Verification uses reference sparse products and never performs the
//! generator's Euclidean column elimination. All work, including admission,
//! hashing and publication, shares one cumulative reference envelope.

use crate::integer_normal_form::{
    IntegerSmithError, MAX_INTEGER_SMITH_COEFFICIENT_BITS, MAX_INTEGER_SMITH_DIMENSION,
    admit_integer_matrix, bounded_integer_matrix, check_inverse_product, multiply,
};
use crate::sparse_rref::{RationalSparseMatrix, Reference, SparseRrefError};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const INTEGER_HERMITE_SCHEMA_VERSION: u32 = 1;
pub const MAX_INTEGER_HERMITE_DIMENSION: usize = MAX_INTEGER_SMITH_DIMENSION;
pub const MAX_INTEGER_HERMITE_COEFFICIENT_BITS: u64 = MAX_INTEGER_SMITH_COEFFICIENT_BITS;

/// Untrusted portable lattice witness. The source matrix is a separate input
/// to verification; no serialized flag can create an accepted receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerHermiteWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub normal_form: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub right: RationalSparseMatrix,
    #[serde(deserialize_with = "bounded_integer_matrix")]
    pub right_inverse: RationalSparseMatrix,
}

/// Accepted complete column-lattice normal form, bound to the exact source
/// and all witness matrices. Digests are commitments, not replay substitutes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerHermiteReceipt {
    rows: usize,
    columns: usize,
    pivot_rows: Vec<usize>,
    source_digest: [u8; 32],
    normal_form_digest: [u8; 32],
    right_digest: [u8; 32],
    right_inverse_digest: [u8; 32],
}

impl IntegerHermiteReceipt {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rank(&self) -> usize {
        self.pivot_rows.len()
    }
    pub fn pivot_rows(&self) -> &[usize] {
        &self.pivot_rows
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn normal_form_digest(&self) -> [u8; 32] {
        self.normal_form_digest
    }
    pub fn right_digest(&self) -> [u8; 32] {
        self.right_digest
    }
    pub fn right_inverse_digest(&self) -> [u8; 32] {
        self.right_inverse_digest
    }
}

#[derive(Debug, Error)]
pub enum IntegerHermiteError {
    #[error("integer Hermite normal form certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

impl From<IntegerSmithError> for IntegerHermiteError {
    fn from(error: IntegerSmithError) -> Self {
        match error {
            IntegerSmithError::Invalid(reason) => Self::Invalid(reason),
            IntegerSmithError::Reference(error) => Self::Reference(error),
        }
    }
}

type Result<T> = std::result::Result<T, IntegerHermiteError>;

/// Check canonical column echelon form without any elimination or rank oracle.
fn check_form(
    form: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Vec<usize>> {
    let mut last: Vec<Option<(usize, &BigRational)>> = work.vector(form.columns)?;
    last.resize(form.columns, None);
    for entry in &form.entries {
        work.tick(1)?;
        // Canonical row-major admission means the last visited entry is the
        // bottommost nonzero in this column, even for noncontiguous pivots.
        last[entry.column] = Some((entry.row, &entry.value));
    }
    let mut pivots = work.vector(form.columns)?;
    for (column, pivot) in last.into_iter().enumerate() {
        work.tick(1)?;
        let Some((row, value)) = pivot else {
            return Err(IntegerHermiteError::Invalid("zero column retained in H"));
        };
        if !value.numer().is_positive() || pivots.last().is_some_and(|previous| *previous >= row) {
            return Err(IntegerHermiteError::Invalid(
                "Hermite pivots must be positive with strictly increasing rows",
            ));
        }
        for entry in &form.entries {
            work.tick(1)?;
            if entry.row == row && entry.column > column {
                work.tick(entry.value.numer().limb_count() + value.numer().limb_count())?;
                // Absent entries represent zero. Stored entries were already
                // admitted as nonzero integers, so positivity is equivalent
                // to the required lower bound here.
                if !entry.value.numer().is_positive() || entry.value.numer() >= value.numer() {
                    return Err(IntegerHermiteError::Invalid(
                        "Hermite pivot-row residue is outside its canonical interval",
                    ));
                }
            }
        }
        pivots.push(row);
    }
    Ok(pivots)
}

/// Independently verify the complete integer column-lattice equality.
/// Rectangular, deficient-rank and zero-area sources are supported. A valid
/// product without integer invertibility, or a canonical-looking H missing
/// a source direction, cannot obtain a receipt.
pub fn verify_integer_hermite(
    source: &RationalSparseMatrix,
    witness: &IntegerHermiteWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<IntegerHermiteReceipt> {
    let mut work = Reference::new(verifier);
    work.tick(1)?;
    if witness.schema_version != INTEGER_HERMITE_SCHEMA_VERSION {
        return Err(IntegerHermiteError::Invalid("unknown witness schema"));
    }
    for matrix in [
        source,
        &witness.normal_form,
        &witness.right,
        &witness.right_inverse,
    ] {
        admit_integer_matrix(matrix, &mut work)?;
    }
    let (m, n) = (source.rows, source.columns);
    let rank = witness.normal_form.columns;
    if witness.normal_form.rows != m
        || rank > m.min(n)
        || [&witness.right, &witness.right_inverse]
            .iter()
            .any(|matrix| (matrix.rows, matrix.columns) != (n, n))
    {
        return Err(IntegerHermiteError::Invalid("Hermite transformation shape"));
    }
    let pivot_rows = check_form(&witness.normal_form, &mut work)?;
    let right = work.rows(&witness.right)?;
    let right_inverse = work.rows(&witness.right_inverse)?;
    check_inverse_product(&right, &right_inverse, &mut work)?;
    check_inverse_product(&right_inverse, &right, &mut work)?;

    let original = work.rows(source)?;
    let product = multiply(&original, &right, n, &mut work)?;
    let omitted = n - rank;
    let mut expected = witness.normal_form.entries.iter();
    for (row, entries) in product.iter().enumerate() {
        for (column, value) in entries {
            work.tick(1)?;
            let Some(entry) = expected.next() else {
                return Err(IntegerHermiteError::Invalid(
                    "A V has an omitted lattice direction",
                ));
            };
            work.tick(value.numer().limb_count() + entry.value.numer().limb_count())?;
            if *column < omitted
                || (row, *column - omitted) != (entry.row, entry.column)
                || value != &entry.value
            {
                return Err(IntegerHermiteError::Invalid("A V differs from [0 | H]"));
            }
        }
    }
    if expected.next().is_some() {
        return Err(IntegerHermiteError::Invalid("unmatched Hermite entries"));
    }
    let source_digest = work.digest(source)?;
    let normal_form_digest = work.digest(&witness.normal_form)?;
    let right_digest = work.digest(&witness.right)?;
    let right_inverse_digest = work.digest(&witness.right_inverse)?;
    work.tick(1)?;
    Ok(IntegerHermiteReceipt {
        rows: m,
        columns: n,
        pivot_rows,
        source_digest,
        normal_form_digest,
        right_digest,
        right_inverse_digest,
    })
}
