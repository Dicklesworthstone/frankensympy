//! Independent inverse and noninvertibility checks over Z/mZ.
//!
//! The original integer A and positive modulus m are supplied separately.
//! A Smith certificate establishes D = U A V over Z with unimodular U,V.
//! An inverse is accepted only after both A X and X A equal I modulo m;
//! noninvertibility requires a nonunit divisor of m and a checked Smith factor.
//! No elimination, modular inverse search, prime-field assumption or floating
//! arithmetic is used by this checker. All phases share one reference budget.

use crate::integer_normal_form::{
    IntegerSmithError, IntegerSmithReceipt, IntegerSmithWitness, admit_integer_matrix,
    verify_integer_smith_with_work,
};
use crate::sparse_rref::{RationalSparseMatrix, Reference, SparseRrefError};
use fsym_budget::{BudgetMeter, MeterError};
use fsym_core::arith::{metered_div_rem, metered_mul, metered_subtract};
use fsym_core::BigInt;
use thiserror::Error;

pub const INTEGER_MODULAR_INVERSE_SCHEMA_VERSION: u32 = 1;
pub const MAX_INTEGER_MODULAR_MODULUS_BITS: u64 = 4096;

/// Proposed residues, or a common divisor proving a Smith diagonal nonunit.
/// A missing diagonal after the checked rank denotes the exact integer zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegerModularInverseOutcome {
    Invertible { inverse: RationalSparseMatrix },
    NonUnit { index: usize, divisor: BigInt },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModularInverseWitness {
    pub schema_version: u32,
    pub normal_form: IntegerSmithWitness,
    pub outcome: IntegerModularInverseOutcome,
}

/// Accepted source, modulus and result binding; no unchecked constructor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModularInverseReceipt {
    normal_form: IntegerSmithReceipt,
    modulus: BigInt,
    inverse_digest: Option<[u8; 32]>,
    nonunit_index: Option<usize>,
}

impl IntegerModularInverseReceipt {
    pub fn normal_form(&self) -> &IntegerSmithReceipt {
        &self.normal_form
    }
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }
    pub fn is_invertible(&self) -> bool {
        self.inverse_digest.is_some()
    }
    pub fn inverse_digest(&self) -> Option<[u8; 32]> {
        self.inverse_digest
    }
    pub fn nonunit_index(&self) -> Option<usize> {
        self.nonunit_index
    }
}

#[derive(Debug, Error)]
pub enum IntegerModularInverseError {
    #[error("integer modular-inverse certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    NormalForm(#[from] IntegerSmithError),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

impl From<MeterError> for IntegerModularInverseError {
    fn from(error: MeterError) -> Self {
        Self::Reference(SparseRrefError::Meter(error))
    }
}

type Result<T> = std::result::Result<T, IntegerModularInverseError>;

fn residue(
    value: &BigInt,
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<BigInt> {
    work.tick(1)?;
    let (_, remainder) = metered_div_rem(value, modulus, work)?
        .ok_or(IntegerModularInverseError::Invalid("zero modulus"))?;
    if remainder.is_negative() {
        // Negating an owned magnitude does not copy its limbs.
        let magnitude = -remainder;
        Ok(metered_subtract(modulus, &magnitude, work)?)
    } else {
        Ok(remainder)
    }
}

fn residues(
    source: &RationalSparseMatrix,
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Vec<BigInt>> {
    let cells = source.rows * source.columns;
    let mut values = work.vector(cells)?;
    work.tick(cells as u64)?;
    values.resize_with(cells, BigInt::zero);
    for entry in &source.entries {
        values[entry.row * source.columns + entry.column] =
            residue(entry.value.numer(), modulus, work)?;
    }
    Ok(values)
}

/// Straight modular dot products are separate from Smith generation and from
/// the generator's V D^-1 U composition. Reduce every term and sum, so an
/// accepted input modulus cannot cause unbounded exact integer intermediates.
fn check_product(
    left: &[BigInt],
    right: &[BigInt],
    size: usize,
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    for row in 0..size {
        for column in 0..size {
            work.storage(8)?;
            let mut sum = BigInt::zero();
            for inner in 0..size {
                work.tick(1)?;
                let a = &left[row * size + inner];
                let b = &right[inner * size + column];
                if a.is_zero() || b.is_zero() {
                    continue;
                }
                let product = metered_mul(a, b, work)?;
                let negative = -residue(&product, modulus, work)?;
                let next = metered_subtract(&sum, &negative, work)?;
                sum = residue(&next, modulus, work)?;
            }
            work.tick(1)?;
            if !(if row == column { sum.is_one() } else { sum.is_zero() }) {
                return Err(IntegerModularInverseError::Invalid(
                    "original modular inverse residual is nonzero",
                ));
            }
        }
    }
    Ok(())
}

/// Independently verify a bounded inverse or a complete noninvertibility claim.
/// Moduli are arbitrary integers >= 2; no primality assumption is made.
pub fn verify_integer_modular_inverse(
    source: &RationalSparseMatrix,
    modulus: &BigInt,
    witness: &IntegerModularInverseWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<IntegerModularInverseReceipt> {
    let mut work = Reference::new(verifier);
    work.tick(1)?;
    if witness.schema_version != INTEGER_MODULAR_INVERSE_SCHEMA_VERSION {
        return Err(IntegerModularInverseError::Invalid("unknown witness schema"));
    }
    if !modulus.is_positive() || modulus.is_one() {
        return Err(IntegerModularInverseError::Invalid("modulus must be at least two"));
    }
    if modulus.bits() > MAX_INTEGER_MODULAR_MODULUS_BITS {
        return Err(SparseRrefError::Limit("modular inverse modulus height").into());
    }
    admit_integer_matrix(source, &mut work)?;
    if source.rows != source.columns {
        return Err(IntegerModularInverseError::Invalid("source must be square"));
    }
    for entry in &source.entries {
        work.tick(1)?;
        if entry.value.numer().bits() > 4096 {
            return Err(SparseRrefError::Limit("modular inverse original coefficient height").into());
        }
    }
    let normal_form =
        verify_integer_smith_with_work(source, &witness.normal_form, &mut work)?;
    let (inverse_digest, nonunit_index) = match &witness.outcome {
        IntegerModularInverseOutcome::Invertible { inverse } => {
            admit_integer_matrix(inverse, &mut work)?;
            if (inverse.rows, inverse.columns) != (source.rows, source.columns) {
                return Err(IntegerModularInverseError::Invalid("inverse shape"));
            }
            for entry in &inverse.entries {
                let value = entry.value.numer();
                work.tick(value.limb_count().max(modulus.limb_count()))?;
                if value.is_negative() || value >= modulus {
                    return Err(IntegerModularInverseError::Invalid(
                        "inverse entries must be canonical modular residues",
                    ));
                }
            }
            let original = residues(source, modulus, &mut work)?;
            let proposed = residues(inverse, modulus, &mut work)?;
            check_product(&original, &proposed, source.rows, modulus, &mut work)?;
            check_product(&proposed, &original, source.rows, modulus, &mut work)?;
            (Some(work.digest(inverse)?), None)
        }
        IntegerModularInverseOutcome::NonUnit { index, divisor } => {
            if *index >= source.rows || !divisor.is_positive() || divisor.is_one() {
                return Err(IntegerModularInverseError::Invalid("nonunit divisor or index"));
            }
            if divisor.bits() > MAX_INTEGER_MODULAR_MODULUS_BITS {
                return Err(SparseRrefError::Limit("nonunit divisor height").into());
            }
            if !residue(modulus, divisor, &mut work)?.is_zero() {
                return Err(IntegerModularInverseError::Invalid(
                    "claimed nonunit does not divide the modulus",
                ));
            }
            if let Some(factor) = witness.normal_form.diagonal.entries.get(*index)
                && !residue(factor.value.numer(), divisor, &mut work)?.is_zero()
            {
                return Err(IntegerModularInverseError::Invalid(
                    "claimed nonunit does not divide the Smith factor",
                ));
            }
            (None, Some(*index))
        }
    };
    work.tick(modulus.limb_count())?;
    work.storage(modulus.limb_count().max(1) * 8)?;
    let modulus = modulus.clone();
    work.tick(1)?;
    Ok(IntegerModularInverseReceipt {
        normal_form,
        modulus,
        inverse_digest,
        nonunit_index,
    })
}
