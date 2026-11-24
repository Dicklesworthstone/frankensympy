//! Complete finite affine families for integer systems A X = B modulo m.
//!
//! The receiver supplies both original matrices and m >= 2. A checked Smith
//! form D = U A V reduces the system to independent scalar congruences.
//! For each nonzero d_i, gcd(d_i,m) must divide (U B)[i,j]; trailing zero
//! rows must vanish modulo m. A solvable witness contains a canonical
//! particular matrix, ordered generators and their exact finite periods.
//! Each RHS column has its own independent parameter vector.
//!
//! Acceptance checks the original residual, canonical V^-1 coordinates and
//! every generator/period against the independently checked unimodular V.
//! Thus the family is complete and has no repeated parameterizations. Merely
//! checking A P = B and A K = 0 would not establish either property.
//! No Smith generation, inverse search, field assumption or floating rank
//! decision enters this checker. All work shares one cumulative reference
//! envelope; interrupted checking publishes no receipt.

use crate::integer_modular_inverse::MAX_INTEGER_MODULAR_MODULUS_BITS;
use crate::integer_normal_form::{
    IntegerSmithError, IntegerSmithReceipt, IntegerSmithWitness, admit_integer_matrix,
    verify_integer_smith_with_work,
};
use crate::sparse_rref::{RationalSparseMatrix, Reference, SparseRrefError};
use fsym_budget::{BudgetMeter, MeterError};
use fsym_core::BigInt;
use fsym_core::arith::{metered_div_rem, metered_gcd, metered_mul, metered_subtract};
use thiserror::Error;

pub const INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION: u32 = 1;
const MAX_ORIGINAL_INTEGER_BITS: u64 = 4096;

/// An untrusted complete family or a transformed divisibility contradiction.
///
/// For a solvable outcome, X = particular + generators*T modulo m, with
/// 0 <= T[i,j] < periods[i], independently for every RHS column j.
/// Only periods greater than one occur, in original Smith-coordinate order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegerModularSystemOutcome {
    Inconsistent {
        row: usize,
        rhs_column: usize,
    },
    Solvable {
        particular: RationalSparseMatrix,
        generators: RationalSparseMatrix,
        periods: Vec<BigInt>,
    },
}

/// Typed replay data; there is no serialized acceptance flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModularSystemWitness {
    pub schema_version: u32,
    pub normal_form: IntegerSmithWitness,
    pub outcome: IntegerModularSystemOutcome,
}

/// Immutable binding of both original sources, modulus and complete outcome.
/// The Smith rank is an integer rank; no field rank over Z/mZ is claimed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModularSystemReceipt {
    normal_form: IntegerSmithReceipt,
    rhs_columns: usize,
    modulus: BigInt,
    contradiction: Option<(usize, usize)>,
    rhs_digest: [u8; 32],
    particular_digest: Option<[u8; 32]>,
    generators_digest: Option<[u8; 32]>,
    periods: Vec<BigInt>,
}

impl IntegerModularSystemReceipt {
    pub fn rows(&self) -> usize {
        self.normal_form.rows()
    }
    pub fn columns(&self) -> usize {
        self.normal_form.columns()
    }
    pub fn rhs_columns(&self) -> usize {
        self.rhs_columns
    }
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }
    pub fn is_consistent(&self) -> bool {
        self.contradiction.is_none()
    }
    /// A zero-column RHS has exactly one empty solution matrix.
    pub fn is_unique(&self) -> bool {
        self.is_consistent() && (self.periods.is_empty() || self.rhs_columns == 0)
    }
    pub fn contradiction(&self) -> Option<(usize, usize)> {
        self.contradiction
    }
    pub fn periods(&self) -> Option<&[BigInt]> {
        self.is_consistent().then_some(self.periods.as_slice())
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
    pub fn generators_digest(&self) -> Option<[u8; 32]> {
        self.generators_digest
    }
    pub fn normal_form(&self) -> &IntegerSmithReceipt {
        &self.normal_form
    }
}

#[derive(Debug, Error)]
pub enum IntegerModularSystemError {
    #[error("integer modular-system certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    NormalForm(#[from] IntegerSmithError),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
}

impl From<MeterError> for IntegerModularSystemError {
    fn from(error: MeterError) -> Self {
        Self::Reference(SparseRrefError::Meter(error))
    }
}

type Result<T> = std::result::Result<T, IntegerModularSystemError>;

fn invalid(reason: &'static str) -> IntegerModularSystemError {
    IntegerModularSystemError::Invalid(reason)
}

fn copy_integer(value: &BigInt, work: &mut Reference<'_, impl BudgetMeter>) -> Result<BigInt> {
    work.tick(value.limb_count().max(1))?;
    work.storage(value.limb_count().max(1) * 8)?;
    Ok(value.clone())
}

fn residue(
    value: &BigInt,
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<BigInt> {
    work.tick(1)?;
    let (_, remainder) =
        metered_div_rem(value, modulus, work)?.ok_or_else(|| invalid("zero modulus"))?;
    if remainder.is_negative() {
        Ok(metered_subtract(modulus, &(-remainder), work)?)
    } else {
        Ok(remainder)
    }
}

fn quotient(
    value: &BigInt,
    divisor: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<BigInt> {
    work.tick(1)?;
    let (quotient, remainder) =
        metered_div_rem(value, divisor, work)?.ok_or_else(|| invalid("zero divisor"))?;
    if !remainder.is_zero() {
        return Err(invalid("nonexact divisor quotient"));
    }
    Ok(quotient)
}

/// Every input has already passed the bounded integer-matrix admission.
struct Residues {
    rows: usize,
    columns: usize,
    values: Vec<BigInt>,
}

impl Residues {
    fn zero(
        rows: usize,
        columns: usize,
        work: &mut Reference<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        let cells = rows * columns;
        let mut values = work.vector(cells)?;
        work.tick(cells as u64)?;
        values.resize_with(cells, BigInt::zero);
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    fn read(
        matrix: &RationalSparseMatrix,
        modulus: &BigInt,
        work: &mut Reference<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        let mut result = Self::zero(matrix.rows, matrix.columns, work)?;
        for entry in &matrix.entries {
            result.values[entry.row * matrix.columns + entry.column] =
                residue(entry.value.numer(), modulus, work)?;
        }
        Ok(result)
    }

    fn get(&self, row: usize, column: usize) -> &BigInt {
        &self.values[row * self.columns + column]
    }

    /// Independent straight dot products reduce each product and sum.
    fn multiply(
        &self,
        right: &Self,
        modulus: &BigInt,
        work: &mut Reference<'_, impl BudgetMeter>,
    ) -> Result<Self> {
        if self.columns != right.rows {
            return Err(invalid("modular product shape"));
        }
        let mut result = Self::zero(self.rows, right.columns, work)?;
        for row in 0..self.rows {
            for column in 0..right.columns {
                let index = row * right.columns + column;
                for inner in 0..self.columns {
                    work.tick(1)?;
                    let left = self.get(row, inner);
                    let right = right.get(inner, column);
                    if left.is_zero() || right.is_zero() {
                        continue;
                    }
                    let product = metered_mul(left, right, work)?;
                    let negative = -residue(&product, modulus, work)?;
                    let sum = metered_subtract(&result.values[index], &negative, work)?;
                    result.values[index] = residue(&sum, modulus, work)?;
                }
            }
        }
        Ok(result)
    }

    fn equal(&self, other: &Self, work: &mut Reference<'_, impl BudgetMeter>) -> Result<bool> {
        if (self.rows, self.columns) != (other.rows, other.columns) {
            return Ok(false);
        }
        for (left, right) in self.values.iter().zip(&other.values) {
            work.tick(left.limb_count().max(right.limb_count()).max(1))?;
            if left != right {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn original(
    matrix: &RationalSparseMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    admit_integer_matrix(matrix, work)?;
    for entry in &matrix.entries {
        work.tick(1)?;
        if entry.value.numer().bits() > MAX_ORIGINAL_INTEGER_BITS {
            return Err(
                SparseRrefError::Limit("modular system original coefficient height").into(),
            );
        }
    }
    Ok(())
}

fn canonical_residues(
    matrix: &RationalSparseMatrix,
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    admit_integer_matrix(matrix, work)?;
    for entry in &matrix.entries {
        let value = entry.value.numer();
        work.tick(value.limb_count().max(modulus.limb_count()))?;
        if value.is_negative() || value >= modulus {
            return Err(invalid("output entries must be canonical modular residues"));
        }
    }
    Ok(())
}

fn admit_outcome(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    modulus: &BigInt,
    outcome: &IntegerModularSystemOutcome,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    work.tick(1)?;
    match outcome {
        IntegerModularSystemOutcome::Inconsistent { row, rhs_column } => {
            if *row >= source.rows || *rhs_column >= rhs.columns {
                return Err(invalid("contradiction coordinates lie outside U B"));
            }
        }
        IntegerModularSystemOutcome::Solvable {
            particular,
            generators,
            periods,
        } => {
            if periods.len() > source.columns {
                return Err(invalid("too many modular parameter periods"));
            }
            for period in periods {
                if period.bits() > MAX_INTEGER_MODULAR_MODULUS_BITS {
                    return Err(SparseRrefError::Limit("modular parameter period height").into());
                }
                work.tick(period.limb_count().max(modulus.limb_count()))?;
                if !period.is_positive() || period.is_one() || period > modulus {
                    return Err(invalid(
                        "parameter periods must lie between two and the modulus",
                    ));
                }
            }
            canonical_residues(particular, modulus, work)?;
            canonical_residues(generators, modulus, work)?;
            if (particular.rows, particular.columns) != (source.columns, rhs.columns)
                || (generators.rows, generators.columns) != (source.columns, periods.len())
            {
                return Err(invalid("modular solution matrix shape"));
            }
        }
    }
    Ok(())
}

fn divisors(
    diagonal: &RationalSparseMatrix,
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Vec<BigInt>> {
    let mut result = work.vector(diagonal.entries.len())?;
    for entry in &diagonal.entries {
        work.tick(1)?;
        result.push(metered_gcd(entry.value.numer(), modulus, work)?);
    }
    Ok(result)
}

fn inconsistent_at(
    transformed_rhs: &Residues,
    divisors: &[BigInt],
    row: usize,
    column: usize,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<bool> {
    work.tick(1)?;
    let value = transformed_rhs.get(row, column);
    if let Some(divisor) = divisors.get(row) {
        Ok(!residue(value, divisor, work)?.is_zero())
    } else {
        Ok(!value.is_zero())
    }
}

fn complete_generators(
    right: &Residues,
    divisors: &[BigInt],
    generators: &Residues,
    periods: &[BigInt],
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    let mut output = 0;
    for coordinate in 0..right.columns {
        work.tick(1)?;
        let period = divisors.get(coordinate).unwrap_or(modulus);
        if period.is_one() {
            continue;
        }
        let supplied = periods
            .get(output)
            .ok_or_else(|| invalid("missing modular generator or period"))?;
        work.tick(period.limb_count().max(supplied.limb_count()))?;
        if period != supplied {
            return Err(invalid("incorrect modular generator period"));
        }
        let step = quotient(modulus, period, work)?;
        for row in 0..right.rows {
            let product = metered_mul(right.get(row, coordinate), &step, work)?;
            let expected = residue(&product, modulus, work)?;
            let proposed = generators.get(row, output);
            work.tick(expected.limb_count().max(proposed.limb_count()))?;
            if expected != *proposed {
                return Err(invalid(
                    "generator differs from its complete Smith direction",
                ));
            }
        }
        output += 1;
    }
    if output != periods.len() {
        return Err(invalid("extra modular generator or period"));
    }
    Ok(())
}

fn canonical_particular(
    coordinates: &Residues,
    divisors: &[BigInt],
    modulus: &BigInt,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    for row in 0..coordinates.rows {
        work.tick(1)?;
        if let Some(divisor) = divisors.get(row) {
            let bound = quotient(modulus, divisor, work)?;
            for column in 0..coordinates.columns {
                let value = coordinates.get(row, column);
                work.tick(value.limb_count().max(bound.limb_count()))?;
                if value >= &bound {
                    return Err(invalid("particular Smith coordinate is not canonical"));
                }
            }
        } else {
            for column in 0..coordinates.columns {
                work.tick(1)?;
                if !coordinates.get(row, column).is_zero() {
                    return Err(invalid("particular free Smith coordinate is not zero"));
                }
            }
        }
    }
    Ok(())
}

/// Verify all solutions modulo an arbitrary integer m >= 2, or inconsistency.
///
/// Original A/B admission, the complete Smith proof, all RHS columns, periods,
/// canonical representatives, residuals and receipt publication share the same
/// 20M-step/256MiB reference envelope. Resource and cancellation errors never
/// become mathematical inconsistency. A rank-zero source still validates every
/// original entry and preserves all finite free directions.
pub fn verify_integer_modular_system(
    source: &RationalSparseMatrix,
    rhs: &RationalSparseMatrix,
    modulus: &BigInt,
    witness: &IntegerModularSystemWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<IntegerModularSystemReceipt> {
    let mut work = Reference::new(verifier);
    work.tick(1)?;
    if witness.schema_version != INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION {
        return Err(invalid("unknown witness schema"));
    }
    if !modulus.is_positive() || modulus.is_one() {
        return Err(invalid("modulus must be at least two"));
    }
    if modulus.bits() > MAX_INTEGER_MODULAR_MODULUS_BITS {
        return Err(SparseRrefError::Limit("modular system modulus height").into());
    }
    original(source, &mut work)?;
    original(rhs, &mut work)?;
    if rhs.rows != source.rows {
        return Err(invalid("original RHS row count"));
    }
    admit_outcome(source, rhs, modulus, &witness.outcome, &mut work)?;
    let normal_form = verify_integer_smith_with_work(source, &witness.normal_form, &mut work)?;
    let factors = divisors(&witness.normal_form.diagonal, modulus, &mut work)?;
    let left = Residues::read(&witness.normal_form.left, modulus, &mut work)?;
    let rhs_values = Residues::read(rhs, modulus, &mut work)?;
    let transformed = left.multiply(&rhs_values, modulus, &mut work)?;

    let (contradiction, particular_digest, generators_digest, accepted_periods) = match &witness
        .outcome
    {
        IntegerModularSystemOutcome::Inconsistent { row, rhs_column } => {
            if !inconsistent_at(&transformed, &factors, *row, *rhs_column, &mut work)? {
                return Err(invalid(
                    "claimed contradiction is a solvable scalar congruence",
                ));
            }
            (Some((*row, *rhs_column)), None, None, work.vector(0)?)
        }
        IntegerModularSystemOutcome::Solvable {
            particular,
            generators,
            periods,
        } => {
            for row in 0..source.rows {
                for column in 0..rhs.columns {
                    if inconsistent_at(&transformed, &factors, row, column, &mut work)? {
                        return Err(invalid("a transformed RHS congruence is inconsistent"));
                    }
                }
            }
            let right = Residues::read(&witness.normal_form.right, modulus, &mut work)?;
            let proposed_generators = Residues::read(generators, modulus, &mut work)?;
            complete_generators(
                &right,
                &factors,
                &proposed_generators,
                periods,
                modulus,
                &mut work,
            )?;
            let proposed = Residues::read(particular, modulus, &mut work)?;
            let inverse = Residues::read(&witness.normal_form.right_inverse, modulus, &mut work)?;
            let coordinates = inverse.multiply(&proposed, modulus, &mut work)?;
            canonical_particular(&coordinates, &factors, modulus, &mut work)?;
            let original = Residues::read(source, modulus, &mut work)?;
            if !original
                .multiply(&proposed, modulus, &mut work)?
                .equal(&rhs_values, &mut work)?
            {
                return Err(invalid("original modular particular residual is nonzero"));
            }
            let mut accepted = work.vector(periods.len())?;
            for period in periods {
                accepted.push(copy_integer(period, &mut work)?);
            }
            (
                None,
                Some(work.digest(particular)?),
                Some(work.digest(generators)?),
                accepted,
            )
        }
    };
    let rhs_digest = work.digest(rhs)?;
    let modulus = copy_integer(modulus, &mut work)?;
    work.tick(1)?;
    Ok(IntegerModularSystemReceipt {
        normal_form,
        rhs_columns: rhs.columns,
        modulus,
        contradiction,
        rhs_digest,
        particular_digest,
        generators_digest,
        periods: accepted_periods,
    })
}
