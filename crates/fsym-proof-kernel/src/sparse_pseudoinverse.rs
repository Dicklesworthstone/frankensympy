//! Independent sparse Moore-Penrose and minimum-norm certificates over Q.
//!
//! The receiver provides the exact original coefficient matrix, and the right
//! hand side for a solve. Acceptance checks all four Penrose identities, not a
//! factorization, pivot list, or claimed rank. The solve additionally checks
//! `X = P B`. These identities establish the unique pseudoinverse and the
//! unique minimum-Euclidean-norm least-squares solution over real rationals.
//!
//! Reference products merge sorted sparse row vectors. This crate has no
//! dependency on the matrix generator, LU, or rank-factorization algorithms.
//! All products, comparison, hashing and receipt publication share one bounded
//! meter. No dense matrix is allocated, including for zero-area inputs. The
//! caller supplies protected verifier resources when its policy requires them.

use crate::sparse_rref::{
    MAX_SPARSE_RREF_ENTRIES, RationalSparseMatrix, Reference, SparseRrefError,
};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SPARSE_PSEUDOINVERSE_SCHEMA_VERSION: u32 = 1;
pub const SPARSE_MINIMUM_NORM_SCHEMA_VERSION: u32 = 1;

/// Untrusted proposal. The original source is a separate verifier input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparsePseudoinverseWitness {
    pub schema_version: u32,
    pub pseudoinverse: RationalSparseMatrix,
}

/// A proposal for the minimum-norm solution of `min_X ||A X - B||²`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMinimumNormWitness {
    pub schema_version: u32,
    pub pseudoinverse: SparsePseudoinverseWitness,
    pub solution: RationalSparseMatrix,
}

/// Accepted four-identity relation. Digests are bindings, not standalone proofs;
/// retain the original source and witness to replay the mathematical claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsePseudoinverseReceipt {
    rows: usize,
    columns: usize,
    source_digest: [u8; 32],
    pseudoinverse_digest: [u8; 32],
}

impl SparsePseudoinverseReceipt {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
    pub fn pseudoinverse_digest(&self) -> [u8; 32] {
        self.pseudoinverse_digest
    }
}

/// Accepted minimum-norm least-squares claim, including its pseudoinverse proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseMinimumNormReceipt {
    pseudoinverse: SparsePseudoinverseReceipt,
    rhs_columns: usize,
    rhs_digest: [u8; 32],
    solution_digest: [u8; 32],
}

impl SparseMinimumNormReceipt {
    pub fn pseudoinverse(&self) -> &SparsePseudoinverseReceipt {
        &self.pseudoinverse
    }
    pub fn rhs_columns(&self) -> usize {
        self.rhs_columns
    }
    pub fn rhs_digest(&self) -> [u8; 32] {
        self.rhs_digest
    }
    pub fn solution_digest(&self) -> [u8; 32] {
        self.solution_digest
    }
}

#[derive(Debug, Error)]
pub enum SparsePseudoinverseError {
    #[error("sparse pseudoinverse certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

type Result<T> = std::result::Result<T, SparsePseudoinverseError>;
type Row = Vec<(usize, BigRational)>;

fn product<M: BudgetMeter>(
    left: &[Row],
    right: &[Row],
    work: &mut Reference<'_, M>,
) -> Result<Vec<Row>> {
    let mut output = work.vector(left.len())?;
    let mut stored = 0;
    for row in left {
        work.tick(1)?;
        let mut result = work.vector(0)?;
        for (inner, coefficient) in row {
            work.tick(1)?;
            let other = right.get(*inner).ok_or(SparsePseudoinverseError::Invalid(
                "product inner coordinate",
            ))?;
            result = work.add_row(
                &result,
                other,
                coefficient,
                MAX_SPARSE_RREF_ENTRIES - stored,
            )?;
        }
        stored += result.len();
        output.push(result);
    }
    Ok(output)
}

fn equal<M: BudgetMeter>(
    rows: &[Row],
    expected: &RationalSparseMatrix,
    message: &'static str,
    work: &mut Reference<'_, M>,
) -> Result<()> {
    if rows.len() != expected.rows {
        return Err(SparsePseudoinverseError::Invalid(message));
    }
    let mut entries = expected.entries.iter();
    for (row, values) in rows.iter().enumerate() {
        work.tick(1)?;
        for (column, value) in values {
            work.tick(value.numer().limb_count() + value.denom().limb_count())?;
            let Some(entry) = entries.next() else {
                return Err(SparsePseudoinverseError::Invalid(message));
            };
            if (entry.row, entry.column) != (row, *column) || &entry.value != value {
                return Err(SparsePseudoinverseError::Invalid(message));
            }
        }
    }
    if entries.next().is_some() {
        return Err(SparsePseudoinverseError::Invalid(message));
    }
    Ok(())
}

fn symmetric<M: BudgetMeter>(
    rows: &[Row],
    message: &'static str,
    work: &mut Reference<'_, M>,
) -> Result<()> {
    for (row, values) in rows.iter().enumerate() {
        work.tick(1)?;
        for (column, value) in values {
            let other = rows
                .get(*column)
                .ok_or(SparsePseudoinverseError::Invalid(message))?;
            work.tick(
                u64::from(usize::BITS - other.len().leading_zeros())
                    + value.numer().limb_count()
                    + value.denom().limb_count(),
            )?;
            let Ok(index) = other.binary_search_by_key(&row, |entry| entry.0) else {
                return Err(SparsePseudoinverseError::Invalid(message));
            };
            if &other[index].1 != value {
                return Err(SparsePseudoinverseError::Invalid(message));
            }
        }
    }
    Ok(())
}

fn verify<M: BudgetMeter>(
    source: &RationalSparseMatrix,
    witness: &SparsePseudoinverseWitness,
    work: &mut Reference<'_, M>,
) -> Result<SparsePseudoinverseReceipt> {
    work.tick(1)?;
    if witness.schema_version != SPARSE_PSEUDOINVERSE_SCHEMA_VERSION {
        return Err(SparsePseudoinverseError::Invalid("unknown witness schema"));
    }
    let inverse = &witness.pseudoinverse;
    work.check_matrix(source)?;
    work.check_matrix(inverse)?;
    if (inverse.rows, inverse.columns) != (source.columns, source.rows) {
        return Err(SparsePseudoinverseError::Invalid("pseudoinverse shape"));
    }
    let a = work.rows(source)?;
    let p = work.rows(inverse)?;
    let ap = product(&a, &p, work)?;
    let pa = product(&p, &a, work)?;
    equal(&product(&ap, &a, work)?, source, "A P A != A", work)?;
    equal(&product(&pa, &p, work)?, inverse, "P A P != P", work)?;
    symmetric(&ap, "A P is not symmetric", work)?;
    symmetric(&pa, "P A is not symmetric", work)?;
    let source_digest = work.digest(source)?;
    let pseudoinverse_digest = work.digest(inverse)?;
    work.tick(1)?;
    Ok(SparsePseudoinverseReceipt {
        rows: source.rows,
        columns: source.columns,
        source_digest,
        pseudoinverse_digest,
    })
}

/// Check all four exact Penrose identities against the caller's original A.
/// Complex, symbolic, approximate and noncanonical source domains are outside
/// this rational schema. Failure or cancellation returns no receipt.
pub fn verify_sparse_pseudoinverse<M: BudgetMeter>(
    source: &RationalSparseMatrix,
    witness: &SparsePseudoinverseWitness,
    meter: &mut M,
) -> Result<SparsePseudoinverseReceipt> {
    verify(source, witness, &mut Reference::new(meter))
}

/// Verify the unique minimum-norm least-squares solution, including all four
/// identities for P and the independent exact product `solution = P * rhs`.
/// An empty RHS does not skip pseudoinverse verification or source admission.
pub fn verify_sparse_minimum_norm<M: BudgetMeter>(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    witness: &SparseMinimumNormWitness,
    meter: &mut M,
) -> Result<SparseMinimumNormReceipt> {
    let mut work = Reference::new(meter);
    work.tick(1)?;
    if witness.schema_version != SPARSE_MINIMUM_NORM_SCHEMA_VERSION {
        return Err(SparsePseudoinverseError::Invalid("unknown solve schema"));
    }
    work.check_matrix(rhs)?;
    work.check_matrix(&witness.solution)?;
    if rhs.rows != source.rows
        || (witness.solution.rows, witness.solution.columns) != (source.columns, rhs.columns)
    {
        return Err(SparsePseudoinverseError::Invalid(
            "minimum-norm solve shape",
        ));
    }
    let pseudoinverse = verify(source, &witness.pseudoinverse, &mut work)?;
    let p = work.rows(&witness.pseudoinverse.pseudoinverse)?;
    let b = work.rows(rhs)?;
    equal(
        &product(&p, &b, &mut work)?,
        &witness.solution,
        "solution != P B",
        &mut work,
    )?;
    let rhs_digest = work.digest(rhs)?;
    let solution_digest = work.digest(&witness.solution)?;
    work.tick(1)?;
    Ok(SparseMinimumNormReceipt {
        pseudoinverse,
        rhs_columns: rhs.columns,
        rhs_digest,
        solution_digest,
    })
}
