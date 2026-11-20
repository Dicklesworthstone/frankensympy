//! Complete integer solutions of `A X = B` from an independently checked
//! Smith form `D = U A V`.
//!
//! With C = U B and X = V Y, each nonzero diagonal equation requires
//! `D[i,i]` to divide every C[i,j]; every remaining row of C must vanish.
//! A cited failure proves inconsistency over Z, including systems that do
//! have rational solutions. Otherwise the canonical particular solution sets
//! free Y coordinates to zero, and the final columns of unimodular V form a
//! complete primitive integer kernel basis. Merely checking A*K=0 and the
//! rational rank of K would not establish this integer completeness claim.
//!
//! The accepted family is `X = particular + kernel * T`, for every integer
//! matrix T of shape `(columns(A)-rank(A)) x columns(B)`. With zero RHS
//! columns there is exactly one empty X even when the kernel is nontrivial.
//! All phases share the caller's meter and one cumulative reference envelope.
//! Callers supply protected verifier resources when required. No matrix
//! generator, optimizer, floating-point rank decision or stored verdict enters
//! acceptance.

use crate::integer_normal_form::{
    IntegerSmithError, IntegerSmithReceipt, IntegerSmithWitness, admit_integer_matrix,
    bounded_integer_matrix, multiply, verify_integer_smith_with_work,
};
use crate::sparse_rref::{RationalSparseMatrix, Reference, SparseRrefError, check_scalar};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use fsym_core::arith::RationalArithmeticError;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const INTEGER_LINEAR_SCHEMA_VERSION: u32 = 1;

/// Untrusted classification of all RHS columns together. A contradiction row
/// names a row of U*B, not a row of the original B.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum IntegerLinearOutcome {
    Inconsistent {
        row: usize,
        rhs_column: usize,
    },
    Solvable {
        #[serde(deserialize_with = "bounded_integer_matrix")]
        particular: RationalSparseMatrix,
        #[serde(deserialize_with = "bounded_integer_matrix")]
        kernel: RationalSparseMatrix,
    },
}

/// Untrusted integer-solution witness. Both original matrices are supplied
/// separately by the receiver and are checked before any outcome shortcut.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerLinearWitness {
    pub schema_version: u32,
    pub normal_form: IntegerSmithWitness,
    pub outcome: IntegerLinearOutcome,
}

/// Source-bound complete integer affine family or integer inconsistency.
/// There is no unchecked constructor or deserializer. Retain the original
/// coefficient/RHS matrices and witness for independent replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerLinearReceipt {
    normal_form: IntegerSmithReceipt,
    rhs_columns: usize,
    contradiction: Option<(usize, usize)>,
    rhs_digest: [u8; 32],
    particular_digest: Option<[u8; 32]>,
    kernel_digest: Option<[u8; 32]>,
}

impl IntegerLinearReceipt {
    pub fn rows(&self) -> usize {
        self.normal_form.rows()
    }
    pub fn columns(&self) -> usize {
        self.normal_form.columns()
    }
    pub fn rhs_columns(&self) -> usize {
        self.rhs_columns
    }
    pub fn rank(&self) -> usize {
        self.normal_form.rank()
    }
    pub fn nullity(&self) -> usize {
        self.columns() - self.rank()
    }
    pub fn is_consistent(&self) -> bool {
        self.contradiction.is_none()
    }
    /// Whether exactly one integer matrix X of the requested shape exists.
    pub fn is_unique(&self) -> bool {
        self.is_consistent() && (self.nullity() == 0 || self.rhs_columns == 0)
    }
    /// `(row in U*B, RHS column)` whose divisibility/zero-row condition fails.
    pub fn contradiction(&self) -> Option<(usize, usize)> {
        self.contradiction
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.normal_form.source_digest()
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
    pub fn normal_form(&self) -> &IntegerSmithReceipt {
        &self.normal_form
    }
}

#[derive(Debug, Error)]
pub enum IntegerLinearError {
    #[error("integer linear-system certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    NormalForm(#[from] IntegerSmithError),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

impl From<RationalArithmeticError> for IntegerLinearError {
    fn from(error: RationalArithmeticError) -> Self {
        Self::Reference(error.into())
    }
}

type Result<T> = std::result::Result<T, IntegerLinearError>;
type Row = Vec<(usize, BigRational)>;

fn validate_outcome_shape(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    outcome: &IntegerLinearOutcome,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    match outcome {
        IntegerLinearOutcome::Inconsistent { row, rhs_column } => {
            if *row >= source.rows || *rhs_column >= rhs.columns {
                return Err(IntegerLinearError::Invalid(
                    "contradiction coordinates lie outside U B",
                ));
            }
        }
        IntegerLinearOutcome::Solvable { particular, kernel } => {
            admit_integer_matrix(particular, work)?;
            admit_integer_matrix(kernel, work)?;
            if (particular.rows, particular.columns) != (source.columns, rhs.columns)
                || kernel.rows != source.columns
            {
                return Err(IntegerLinearError::Invalid("integer solution matrix shape"));
            }
        }
    }
    Ok(())
}

fn check_contradiction(
    transformed_rhs: &[Row],
    diagonal: &RationalSparseMatrix,
    row: usize,
    rhs_column: usize,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    let entries = &transformed_rhs[row];
    work.tick(entries.len() as u64)?;
    let position = entries
        .binary_search_by_key(&rhs_column, |(column, _)| *column)
        .map_err(|_| IntegerLinearError::Invalid("claimed contradiction is zero"))?;
    let value = &entries[position].1;
    // The independently checked canonical D stores exactly its nonzero
    // leading diagonal. Every omitted row is an exact zero equation.
    if let Some(divisor) = diagonal.entries.get(row) {
        work.tick(1)?;
        if value.metered_rem(&divisor.value, work)?.is_zero() {
            return Err(IntegerLinearError::Invalid(
                "claimed contradiction satisfies integer divisibility",
            ));
        }
    }
    Ok(())
}

fn diagonal_particular(
    transformed_rhs: &[Row],
    diagonal: &RationalSparseMatrix,
    variable_count: usize,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Vec<Row>> {
    let rank = diagonal.entries.len();
    for row in &transformed_rhs[rank..] {
        work.tick(1)?;
        if !row.is_empty() {
            return Err(IntegerLinearError::Invalid(
                "a zero Smith row has a nonzero transformed right-hand side",
            ));
        }
    }
    let mut coordinates = work.vector(variable_count)?;
    for (index, divisor) in diagonal.entries.iter().enumerate() {
        work.tick(1)?;
        let mut row = work.vector(transformed_rhs[index].len())?;
        for (column, value) in &transformed_rhs[index] {
            work.tick(1)?;
            let quotient = value.metered_div(&divisor.value, work)?;
            check_scalar(&quotient)?;
            if !quotient.is_integer() {
                return Err(IntegerLinearError::Invalid(
                    "transformed right-hand side is not divisible by a Smith factor",
                ));
            }
            if !quotient.is_zero() {
                row.push((*column, quotient));
            }
        }
        coordinates.push(row);
    }
    for _ in rank..variable_count {
        coordinates.push(work.vector(0)?);
    }
    Ok(coordinates)
}

fn check_particular(
    expected: &[Row],
    supplied: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    let mut entries = supplied.entries.iter();
    for (row, values) in expected.iter().enumerate() {
        work.tick(1)?;
        for (column, value) in values {
            let entry = entries.next().ok_or(IntegerLinearError::Invalid(
                "truncated canonical integer particular solution",
            ))?;
            work.tick(value.numer().limb_count() + entry.value.numer().limb_count())?;
            if (entry.row, entry.column) != (row, *column) || entry.value != *value {
                return Err(IntegerLinearError::Invalid(
                    "particular solution differs from V times the exact diagonal solution",
                ));
            }
        }
    }
    if entries.next().is_some() {
        return Err(IntegerLinearError::Invalid(
            "extra entries in canonical integer particular solution",
        ));
    }
    Ok(())
}

fn check_complete_kernel(
    right: &RationalSparseMatrix,
    rank: usize,
    kernel: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    if kernel.columns != right.columns - rank {
        return Err(IntegerLinearError::Invalid("integer kernel nullity"));
    }
    let mut supplied = kernel.entries.iter();
    for entry in &right.entries {
        work.tick(1)?;
        if entry.column < rank {
            continue;
        }
        let candidate = supplied.next().ok_or(IntegerLinearError::Invalid(
            "truncated integer kernel basis",
        ))?;
        work.tick(entry.value.numer().limb_count() + candidate.value.numer().limb_count())?;
        if (candidate.row, candidate.column) != (entry.row, entry.column - rank)
            || candidate.value != entry.value
        {
            return Err(IntegerLinearError::Invalid(
                "kernel differs from the complete integer basis in V",
            ));
        }
    }
    if supplied.next().is_some() {
        return Err(IntegerLinearError::Invalid("extra integer kernel entries"));
    }
    Ok(())
}

/// Verify either the complete integer family `X=P+K*T` or integer
/// inconsistency for the separately supplied original `A*X=B`.
///
/// The Smith proof, transformed RHS, exact divisions, primitive kernel,
/// particular solution, digests and publication all share the same intrinsic
/// reference envelope. Unknown schemas, malformed candidates, operational
/// failure and cancellation never become a no-solution or accepted verdict.
pub fn verify_integer_linear_system(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    witness: &IntegerLinearWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<IntegerLinearReceipt> {
    let mut work = Reference::new(verifier);
    work.tick(1)?;
    if witness.schema_version != INTEGER_LINEAR_SCHEMA_VERSION {
        return Err(IntegerLinearError::Invalid("unknown witness schema"));
    }
    admit_integer_matrix(source, &mut work)?;
    admit_integer_matrix(rhs, &mut work)?;
    if rhs.rows != source.rows {
        return Err(IntegerLinearError::Invalid("original RHS row count"));
    }
    validate_outcome_shape(source, rhs, &witness.outcome, &mut work)?;
    let normal_form = verify_integer_smith_with_work(source, &witness.normal_form, &mut work)?;
    let left = work.rows(&witness.normal_form.left)?;
    let rhs_rows = work.rows(rhs)?;
    let transformed_rhs = multiply(&left, &rhs_rows, rhs.columns, &mut work)?;

    let (contradiction, particular_digest, kernel_digest) = match &witness.outcome {
        IntegerLinearOutcome::Inconsistent { row, rhs_column } => {
            check_contradiction(
                &transformed_rhs,
                &witness.normal_form.diagonal,
                *row,
                *rhs_column,
                &mut work,
            )?;
            (Some((*row, *rhs_column)), None, None)
        }
        IntegerLinearOutcome::Solvable { particular, kernel } => {
            let coordinates = diagonal_particular(
                &transformed_rhs,
                &witness.normal_form.diagonal,
                source.columns,
                &mut work,
            )?;
            check_complete_kernel(
                &witness.normal_form.right,
                normal_form.rank(),
                kernel,
                &mut work,
            )?;
            let right = work.rows(&witness.normal_form.right)?;
            let expected = multiply(&right, &coordinates, rhs.columns, &mut work)?;
            check_particular(&expected, particular, &mut work)?;
            (
                None,
                Some(work.digest(particular)?),
                Some(work.digest(kernel)?),
            )
        }
    };
    let rhs_digest = work.digest(rhs)?;
    work.tick(1)?;
    Ok(IntegerLinearReceipt {
        normal_form,
        rhs_columns: rhs.columns,
        contradiction,
        rhs_digest,
        particular_digest,
        kernel_digest,
    })
}
