//! Independent exact inertia certificates for rational quadratic forms.
//!
//! Acceptance checks `P(A+A^T)P^T = L D L^T` with a genuine permutation P
//! and unit lower triangular L. Congruence preserves the inertia of the
//! symmetric part. The omitted positive factor 1/2 cannot change the signs.
//! Scalar blocks and [[0,b],[b,0]] blocks of D determine the complete inertia;
//! each nonzero paired block has exactly one positive and one negative direction.
//! No matrix generator, pivot search or approximate eigensolver is used.

use crate::sparse_rref::{RationalSparseMatrix, Reference, SparseRrefError};
use fsym_budget::BudgetMeter;
use fsym_core::BigRational;
use num_traits::{One, Signed, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use thiserror::Error;

pub const RATIONAL_INERTIA_SCHEMA_VERSION: u32 = 1;
pub const MAX_RATIONAL_INERTIA_DIMENSION: usize = 64;

/// Untrusted block-LDL congruence. `permutation[i]` names the original
/// coordinate occupying row/column i. D has scalar or zero-diagonal 2x2 blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalInertiaWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_permutation")]
    pub permutation: Vec<usize>,
    pub lower: RationalSparseMatrix,
    pub diagonal: RationalSparseMatrix,
}

/// Complete inertia of x^T A x, not eigenvalue counts of a nonsymmetric A.
/// Retain the original source and witness for independent replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalInertiaReceipt {
    positive: usize,
    negative: usize,
    zero: usize,
    source_digest: [u8; 32],
}
impl RationalInertiaReceipt {
    pub fn positive(&self) -> usize {
        self.positive
    }
    pub fn negative(&self) -> usize {
        self.negative
    }
    pub fn zero(&self) -> usize {
        self.zero
    }
    pub fn dimension(&self) -> usize {
        self.positive + self.negative + self.zero
    }
    pub fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }
}

#[derive(Debug, Error)]
pub enum RationalInertiaError {
    #[error("rational inertia certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}
type Result<T> = std::result::Result<T, RationalInertiaError>;
type Row = Vec<(usize, BigRational)>;

fn bounded_permutation<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<usize>, D::Error> {
    struct Bounded;
    impl<'de> Visitor<'de> for Bounded {
        type Value = Vec<usize>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "at most {MAX_RATIONAL_INERTIA_DIMENSION} coordinate indices"
            )
        }
        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            if seq
                .size_hint()
                .is_some_and(|n| n > MAX_RATIONAL_INERTIA_DIMENSION)
            {
                return Err(serde::de::Error::custom("inertia permutation length"));
            }
            let mut values = Vec::new();
            loop {
                if values.len() == MAX_RATIONAL_INERTIA_DIMENSION {
                    if seq.next_element::<IgnoredAny>()?.is_some() {
                        return Err(serde::de::Error::custom("inertia permutation length"));
                    }
                    return Ok(values);
                }
                let Some(value) = seq.next_element()? else {
                    return Ok(values);
                };
                values.push(value);
            }
        }
    }
    d.deserialize_seq(Bounded)
}

fn transpose(rows: &[Row], work: &mut Reference<'_, impl BudgetMeter>) -> Result<Vec<Row>> {
    let n = rows.len();
    let mut result = work.vector(n)?;
    for _ in 0..n {
        result.push(work.vector(n)?);
    }
    for (row, values) in rows.iter().enumerate() {
        for (column, value) in values {
            work.tick(1)?;
            result[*column].push((row, work.clone_value(value)?));
        }
    }
    Ok(result)
}

// Reference dot products never choose pivots or update Schur complements.
fn dot(left: &Row, right: &Row, work: &mut Reference<'_, impl BudgetMeter>) -> Result<BigRational> {
    work.storage(64)?;
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

/// Check against the original source. Empty square matrices have inertia
/// (0,0,0). Every admission, product, digest and publication shares one meter
/// and the sparse reference lane's 20M-step/256MiB cumulative envelope.
pub fn verify_rational_inertia(
    source: &RationalSparseMatrix,
    witness: &RationalInertiaWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<RationalInertiaReceipt> {
    let mut work = Reference::new(verifier);
    work.tick(1)?;
    if witness.schema_version != RATIONAL_INERTIA_SCHEMA_VERSION {
        return Err(RationalInertiaError::Invalid("unknown witness schema"));
    }
    let n = source.rows;
    if source.columns != n || n > MAX_RATIONAL_INERTIA_DIMENSION {
        return Err(RationalInertiaError::Invalid(
            "square dimension exceeds inertia envelope",
        ));
    }
    for matrix in [source, &witness.lower, &witness.diagonal] {
        if matrix.rows != n || matrix.columns != n || matrix.entries.len() > n * n {
            return Err(RationalInertiaError::Invalid("congruence matrix shape"));
        }
        work.check_matrix(matrix)?;
    }
    if witness.permutation.len() != n {
        return Err(RationalInertiaError::Invalid("permutation dimension"));
    }
    let mut seen = work.vector(n)?;
    seen.resize(n, false);
    for &index in &witness.permutation {
        work.tick(1)?;
        if index >= n || seen[index] {
            return Err(RationalInertiaError::Invalid(
                "permutation is not bijective",
            ));
        }
        seen[index] = true;
    }
    let lower = work.rows(&witness.lower)?;
    for (row, values) in lower.iter().enumerate() {
        work.tick(1)?;
        if !values
            .last()
            .is_some_and(|(column, value)| *column == row && value.is_one())
        {
            return Err(RationalInertiaError::Invalid(
                "L is not unit lower triangular",
            ));
        }
    }
    let diagonal = work.rows(&witness.diagonal)?;
    let (mut positive, mut negative, mut zero, mut k) = (0, 0, 0, 0);
    while k < n {
        work.tick(1)?;
        let row = &diagonal[k];
        if row.is_empty() {
            zero += 1;
            k += 1;
            continue;
        }
        if row.len() == 1 && row[0].0 == k {
            if row[0].1.is_positive() {
                positive += 1;
            } else {
                negative += 1;
            }
            k += 1;
            continue;
        }
        if row.len() == 1 && row[0].0 == k + 1 && k + 1 < n {
            let next = &diagonal[k + 1];
            work.tick(row[0].1.numer().limb_count() + row[0].1.denom().limb_count())?;
            if next.len() == 1 && next[0].0 == k && next[0].1 == row[0].1 {
                positive += 1;
                negative += 1;
                k += 2;
                continue;
            }
        }
        return Err(RationalInertiaError::Invalid(
            "D has invalid scalar or paired blocks",
        ));
    }
    let original = work.rows(source)?;
    let original_t = transpose(&original, &mut work)?;
    work.storage(64)?;
    let one = BigRational::one();
    let mut symmetric = work.vector(n)?;
    for row in 0..n {
        symmetric.push(work.add_row(&original[row], &original_t[row], &one, n)?);
    }
    let diagonal_t = transpose(&diagonal, &mut work)?;
    let mut ld = work.vector(n)?;
    for row in &lower {
        let mut values = work.vector(n)?;
        for (column, right) in diagonal_t.iter().enumerate() {
            let value = dot(row, right, &mut work)?;
            if !value.is_zero() {
                values.push((column, value));
            }
        }
        ld.push(values);
    }
    for (row, left) in ld.iter().enumerate() {
        let original_row = &symmetric[witness.permutation[row]];
        for (column, right) in lower.iter().enumerate() {
            let value = dot(left, right, &mut work)?;
            let original_column = witness.permutation[column];
            let matches = match original_row.binary_search_by_key(&original_column, |(i, _)| *i) {
                Ok(index) => {
                    work.tick(value.numer().limb_count() + value.denom().limb_count())?;
                    value == original_row[index].1
                }
                Err(_) => value.is_zero(),
            };
            if !matches {
                return Err(RationalInertiaError::Invalid("P(A+A^T)P^T != L D L^T"));
            }
        }
    }
    let source_digest = work.digest(source)?;
    work.tick(1)?;
    Ok(RationalInertiaReceipt {
        positive,
        negative,
        zero,
        source_digest,
    })
}
