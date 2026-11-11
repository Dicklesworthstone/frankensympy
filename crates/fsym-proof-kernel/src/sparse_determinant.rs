//! Portable exact determinants bound to original sparse rational matrices.
//!
//! The receiver supplies the original matrix. The independent sparse checker
//! replays invertible row operations and proves the complete RREF. A deficient
//! rank proves determinant zero; a full-rank square RREF is the identity. In
//! that case the claimed determinant, multiplied by every row scale and the
//! sign of every nontrivial swap, must equal one. Row additions do not change
//! determinants. No pivot search, matrix generator, or floating point enters
//! acceptance.
//!
//! Replay, determinant arithmetic, source hashing and receipt publication use
//! one cumulative reference meter and the sparse RREF intrinsic limits. No
//! dense matrix is allocated. The caller supplies protected verifier resources
//! when its execution policy requires them. This module does not reserve them.

use crate::linear_system::LinearRowOperation;
use crate::sparse_rref::{
    RationalSparseMatrix, Reference, SparseRrefError, SparseRrefReceipt, SparseRrefWitness,
    bounded_scalar, check_scalar, verify_sparse_rref_reference,
};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SPARSE_DETERMINANT_SCHEMA_VERSION: u32 = 1;

/// An untrusted portable proposal, not evidence until independently replayed.
/// The expected original matrix is a separate verifier input, never inferred
/// from the witness or a stored digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseDeterminantWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_scalar")]
    pub determinant: BigRational,
    pub reduction: SparseRrefWitness,
}

/// Accepted exact determinant and source-bound row-reduction evidence.
/// There is no unchecked constructor or deserializer. Retain the source and
/// witness for replay; a digest alone never proves this claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseDeterminantReceipt {
    determinant: BigRational,
    reduction: SparseRrefReceipt,
}

impl SparseDeterminantReceipt {
    pub fn determinant(&self) -> &BigRational {
        &self.determinant
    }

    pub fn dimension(&self) -> usize {
        self.reduction.rows()
    }

    pub fn rank(&self) -> usize {
        self.reduction.rank()
    }

    pub fn source_digest(&self) -> [u8; 32] {
        self.reduction.source_digest()
    }

    pub fn reduction(&self) -> &SparseRrefReceipt {
        &self.reduction
    }
}

#[derive(Debug, Error)]
pub enum SparseDeterminantError {
    #[error("sparse determinant certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reduction(#[from] SparseRrefError),
}

/// Check the exact claim `det(source) = witness.determinant` over Q.
/// This proves neither a symbolic-domain extension nor any numerical accuracy
/// statement. Unknown schemas, invalid reductions, exhausted work and
/// cancellation produce no accepted receipt.
pub fn verify_sparse_determinant<M: BudgetMeter>(
    source: &RationalSparseMatrix,
    witness: &SparseDeterminantWitness,
    meter: &mut M,
) -> Result<SparseDeterminantReceipt, SparseDeterminantError> {
    let mut work = Reference::new(meter);
    work.tick(1)?;
    if witness.schema_version != SPARSE_DETERMINANT_SCHEMA_VERSION {
        return Err(SparseDeterminantError::Invalid("unknown witness schema"));
    }
    if source.rows != source.columns {
        return Err(SparseDeterminantError::Invalid("matrix must be square"));
    }
    check_scalar(&witness.determinant)?;

    // Never trust the claimed rank, even when the claimed determinant is zero.
    // Reuse the same intrinsic accounting envelope through final publication.
    let reduction = verify_sparse_rref_reference(source, &witness.reduction, &mut work)?;
    if reduction.rank() < source.rows {
        if !witness.determinant.is_zero() {
            return Err(SparseDeterminantError::Invalid(
                "rank-deficient matrix has nonzero claimed determinant",
            ));
        }
    } else {
        let mut transformed = work.clone_value(&witness.determinant)?;
        let mut negative = false;
        for operation in &witness.reduction.operations {
            work.tick(1)?;
            match operation {
                LinearRowOperation::Swap { left, right } => negative ^= left != right,
                LinearRowOperation::Scale { factor, .. } => {
                    transformed = work.multiply(&transformed, factor)?;
                }
                LinearRowOperation::Add { .. } => {}
            }
        }
        if negative {
            transformed = work.multiply(&transformed, &BigRational::from_integer((-1).into()))?;
        }
        work.tick(transformed.numer().limb_count() + transformed.denom().limb_count())?;
        if !transformed.is_one() {
            return Err(SparseDeterminantError::Invalid(
                "row-operation determinant identity does not hold",
            ));
        }
    }
    let determinant = work.clone_value(&witness.determinant)?;
    work.tick(1)?;
    Ok(SparseDeterminantReceipt {
        determinant,
        reduction,
    })
}
