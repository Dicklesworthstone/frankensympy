//! Independent certificates for the complete solution set of `A x = b` over Q.
//!
//! The witness is a bounded sequence of invertible elementary row operations.
//! This checker replays those operations, checks reduced row echelon form of
//! the coefficient block, and checks every free direction of the reported
//! affine space. A residual alone never establishes completeness or uniqueness.
//! No matrix generator, pivot search, symbolic simplifier, or floating point
//! operation participates in acceptance.

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::marker::PhantomData;
use thiserror::Error;

pub const LINEAR_SYSTEM_SCHEMA_VERSION: u32 = 1;
pub const MAX_LINEAR_SYSTEM_DIMENSION: usize = 128;
pub const MAX_LINEAR_SYSTEM_COEFFICIENT_BITS: u64 = 8_192;
pub const MAX_LINEAR_SYSTEM_OPERATIONS: usize = 32_768;
const MAX_ENTRIES: usize = MAX_LINEAR_SYSTEM_DIMENSION * (MAX_LINEAR_SYSTEM_DIMENSION + 1);
const MAX_WORK: u64 = 20_000_000;
const MAX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;

/// Exact, untrusted claim data. Coefficients are row major; the right-hand
/// side contains one entry per row. Empty dimensions have their usual linear
/// algebra meaning, including the unique empty vector when there are no unknowns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalLinearSystem {
    pub rows: usize,
    pub columns: usize,
    #[serde(deserialize_with = "bounded_entries")]
    pub coefficients: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_values")]
    pub rhs: Vec<BigRational>,
}

/// Every admitted operation is invertible: scale factors are nonzero and
/// addition uses two distinct rows. All indices refer to the current rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum LinearRowOperation {
    Swap {
        left: usize,
        right: usize,
    },
    Scale {
        row: usize,
        #[serde(deserialize_with = "bounded_scalar")]
        factor: BigRational,
    },
    Add {
        target: usize,
        source: usize,
        #[serde(deserialize_with = "bounded_scalar")]
        factor: BigRational,
    },
}

/// A complete solution set, never merely a list of checked candidate vectors.
/// The affine basis has one vector for each increasing `free_columns` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum RationalLinearSolution {
    Unique {
        #[serde(deserialize_with = "bounded_values")]
        values: Vec<BigRational>,
    },
    Affine {
        #[serde(deserialize_with = "bounded_values")]
        particular: Vec<BigRational>,
        #[serde(deserialize_with = "bounded_basis")]
        basis: Vec<Vec<BigRational>>,
        #[serde(deserialize_with = "bounded_columns")]
        free_columns: Vec<usize>,
    },
    Inconsistent {
        row: usize,
    },
}

/// Portable untrusted witness. `reduced` is the flattened augmented matrix
/// `[A | b]` after the row operations; `pivots` describe only the A block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearSystemWitness {
    pub schema_version: u32,
    #[serde(deserialize_with = "bounded_operations")]
    pub operations: Vec<LinearRowOperation>,
    #[serde(deserialize_with = "bounded_entries")]
    pub reduced: Vec<BigRational>,
    #[serde(deserialize_with = "bounded_columns")]
    pub pivots: Vec<usize>,
    pub solution: RationalLinearSolution,
}

/// Successful check metadata. The original claim and witness must be retained
/// with this receipt; the receipt is not a substitute for either artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearSystemVerificationReceipt {
    rank: usize,
    nullity: usize,
    consistent: bool,
    operations: usize,
}

impl LinearSystemVerificationReceipt {
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn nullity(&self) -> usize {
        self.nullity
    }
    pub fn is_consistent(&self) -> bool {
        self.consistent
    }
    pub fn row_operations(&self) -> usize {
        self.operations
    }
}

#[derive(Debug, Error)]
pub enum LinearSystemVerificationError {
    #[error("linear system certificate rejected: {0}")]
    Invalid(&'static str),
    #[error("linear system verification resource limit: {0}")]
    Limit(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for LinearSystemVerificationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, LinearSystemVerificationError>;
fn invalid(message: &'static str) -> LinearSystemVerificationError {
    LinearSystemVerificationError::Invalid(message)
}

// Check each integer before rational normalization and each scalar before the
// next sequence element is retained. The foundational BigInt decoder bounds
// one wire integer; an in-bound count of oversized integers must not accumulate
// across an otherwise bounded matrix or row-operation sequence.
struct BoundedInteger(BigInt);

impl<'de> Deserialize<'de> for BoundedInteger {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let value = BigInt::deserialize(d)?;
        if value.bits() > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS {
            return Err(serde::de::Error::custom("linear coefficient height limit"));
        }
        Ok(Self(value))
    }
}

fn bounded_scalar<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<BigRational, D::Error> {
    let (BoundedInteger(numerator), BoundedInteger(denominator)) =
        <(BoundedInteger, BoundedInteger)>::deserialize(d)?;
    if !denominator.is_positive() || !numerator.gcd(&denominator).is_one() {
        return Err(serde::de::Error::custom(
            "noncanonical linear rational coefficient",
        ));
    }
    Ok(BigRational::new(numerator, denominator))
}

#[derive(Deserialize)]
struct BoundedScalar(#[serde(deserialize_with = "bounded_scalar")] BigRational);

fn bounded_vec<'de, D, T, const MAX: usize>(
    deserializer: D,
) -> std::result::Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Bounded<T, const MAX: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const MAX: usize> Visitor<'de> for Bounded<T, MAX> {
        type Value = Vec<T>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "at most {MAX} certificate entries")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vec<T>, A::Error> {
            if seq.size_hint().is_some_and(|size| size > MAX) {
                return Err(serde::de::Error::custom(
                    "linear certificate sequence limit",
                ));
            }
            let mut values = Vec::new();
            loop {
                if values.len() == MAX {
                    if seq.next_element::<IgnoredAny>()?.is_some() {
                        return Err(serde::de::Error::custom(
                            "linear certificate sequence limit",
                        ));
                    }
                    return Ok(values);
                }
                match seq.next_element()? {
                    Some(value) => values.push(value),
                    None => return Ok(values),
                }
            }
        }
    }
    deserializer.deserialize_seq(Bounded::<T, MAX>(PhantomData))
}

fn bounded_entries<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<BigRational>, D::Error> {
    bounded_vec::<D, BoundedScalar, MAX_ENTRIES>(d)
        .map(|values| values.into_iter().map(|value| value.0).collect())
}
fn bounded_values<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<BigRational>, D::Error> {
    bounded_vec::<D, BoundedScalar, MAX_LINEAR_SYSTEM_DIMENSION>(d)
        .map(|values| values.into_iter().map(|value| value.0).collect())
}
fn bounded_columns<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<usize>, D::Error> {
    bounded_vec::<D, usize, MAX_LINEAR_SYSTEM_DIMENSION>(d)
}
fn bounded_operations<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<LinearRowOperation>, D::Error> {
    bounded_vec::<D, LinearRowOperation, MAX_LINEAR_SYSTEM_OPERATIONS>(d)
}
fn bounded_basis<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Vec<Vec<BigRational>>, D::Error> {
    #[derive(Deserialize)]
    struct Row(#[serde(deserialize_with = "bounded_values")] Vec<BigRational>);
    bounded_vec::<D, Row, MAX_LINEAR_SYSTEM_DIMENSION>(d)
        .map(|rows| rows.into_iter().map(|row| row.0).collect())
}

fn check_scalar(value: &BigRational) -> Result<()> {
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(invalid("invalid rational denominator"));
    }
    if value.numer().bits().max(value.denom().bits()) > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS {
        return Err(LinearSystemVerificationError::Limit("coefficient height"));
    }
    Ok(())
}

/// Counts actual substrate charges as well as containers. These are logical
/// cumulative allocation bounds, not measurements of the process allocator.
struct Reference<'a, M> {
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

impl<M: BudgetMeter> Reference<'_, M> {
    fn tick(&mut self, steps: usize) -> Result<()> {
        self.checkpoint()?;
        self.charge(Dimension::ComputeSteps, steps.max(1) as u64)?;
        Ok(())
    }
    fn storage(&mut self, bytes: usize) -> Result<()> {
        self.tick(1)?;
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }
    fn clone_value(&mut self, value: &BigRational) -> Result<BigRational> {
        check_scalar(value)?;
        self.storage(
            ((value.numer().limb_count() + value.denom().limb_count()).max(1) * 8) as usize,
        )?;
        Ok(value.clone())
    }
    fn multiply(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        if a.numer().bits().saturating_add(b.numer().bits()) > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS
            || a.denom().bits().saturating_add(b.denom().bits())
                > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS
        {
            return Err(LinearSystemVerificationError::Limit(
                "multiplication intermediate",
            ));
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
        if numerator > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS
            || a.denom().bits().saturating_add(b.denom().bits())
                > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS
        {
            return Err(LinearSystemVerificationError::Limit(
                "addition intermediate",
            ));
        }
        self.tick(1)?;
        let result = a.metered_add(b, self)?;
        check_scalar(&result)?;
        Ok(result)
    }
}

fn preflight<M: BudgetMeter>(
    system: &RationalLinearSystem,
    witness: &LinearSystemWitness,
    work: &mut Reference<'_, M>,
) -> Result<()> {
    work.tick(1)?;
    let (rows, columns) = (system.rows, system.columns);
    if rows > MAX_LINEAR_SYSTEM_DIMENSION || columns > MAX_LINEAR_SYSTEM_DIMENSION {
        return Err(LinearSystemVerificationError::Limit("matrix dimensions"));
    }
    if system.coefficients.len() != rows * columns || system.rhs.len() != rows {
        return Err(invalid("claim shape"));
    }
    if witness.schema_version != LINEAR_SYSTEM_SCHEMA_VERSION
        || witness.reduced.len() != rows * (columns + 1)
        || witness.pivots.len() > rows.min(columns)
        || witness.operations.len() > MAX_LINEAR_SYSTEM_OPERATIONS
    {
        return Err(invalid("witness schema or shape"));
    }
    let values: &[BigRational] = match &witness.solution {
        RationalLinearSolution::Unique { values } => {
            if values.len() != columns {
                return Err(invalid("unique solution length"));
            }
            values
        }
        RationalLinearSolution::Affine {
            particular,
            basis,
            free_columns,
        } => {
            if particular.len() != columns
                || basis.is_empty()
                || basis.len() > columns
                || free_columns.len() != basis.len()
                || basis.iter().any(|v| v.len() != columns)
            {
                return Err(invalid("affine solution shape"));
            }
            for vector in basis {
                for value in vector {
                    work.tick(1)?;
                    check_scalar(value)?;
                }
            }
            particular
        }
        RationalLinearSolution::Inconsistent { row } => {
            if *row >= rows {
                return Err(invalid("inconsistency row index"));
            }
            &[]
        }
    };
    for value in system
        .coefficients
        .iter()
        .chain(&system.rhs)
        .chain(&witness.reduced)
        .chain(values)
    {
        work.tick(1)?;
        check_scalar(value)?;
    }
    for operation in &witness.operations {
        work.tick(1)?;
        match operation {
            LinearRowOperation::Swap { left, right } => {
                if *left >= rows || *right >= rows || left == right {
                    return Err(invalid("invalid row swap"));
                }
            }
            LinearRowOperation::Scale { row, factor } => {
                check_scalar(factor)?;
                if *row >= rows || factor.is_zero() {
                    return Err(invalid("noninvertible row scaling"));
                }
            }
            LinearRowOperation::Add {
                target,
                source,
                factor,
            } => {
                check_scalar(factor)?;
                if *target >= rows || *source >= rows || target == source {
                    return Err(invalid("invalid row addition"));
                }
            }
        }
    }
    Ok(())
}

/// Verify the exact rational claim and the entire solution set using the
/// caller's verifier meter. Allocation and arithmetic are charged before work;
/// cancellation or exhaustion returns no receipt. The caller can supply a
/// meter backed by a protected verifier lease independently of the generator's
/// budget.
pub fn verify_linear_system_witness<M: BudgetMeter>(
    system: &RationalLinearSystem,
    witness: &LinearSystemWitness,
    meter: &mut M,
) -> Result<LinearSystemVerificationReceipt> {
    let mut work = Reference {
        meter,
        steps: 0,
        bytes: 0,
    };
    preflight(system, witness, &mut work)?;
    let (rows, columns, width) = (system.rows, system.columns, system.columns + 1);
    work.storage(rows * width * std::mem::size_of::<BigRational>())?;
    let mut augmented = Vec::with_capacity(rows * width);
    for row in 0..rows {
        for column in 0..columns {
            augmented.push(work.clone_value(&system.coefficients[row * columns + column])?);
        }
        augmented.push(work.clone_value(&system.rhs[row])?);
    }
    for operation in &witness.operations {
        work.tick(width)?;
        match operation {
            LinearRowOperation::Swap { left, right } => {
                for column in 0..width {
                    augmented.swap(left * width + column, right * width + column);
                }
            }
            LinearRowOperation::Scale { row, factor } => {
                for column in 0..width {
                    let index = row * width + column;
                    augmented[index] = work.multiply(&augmented[index], factor)?;
                }
            }
            LinearRowOperation::Add {
                target,
                source,
                factor,
            } => {
                for column in 0..width {
                    let index = target * width + column;
                    let term = work.multiply(&augmented[source * width + column], factor)?;
                    augmented[index] = work.add(&augmented[index], &term)?;
                }
            }
        }
    }
    work.tick(augmented.len())?;
    if augmented != witness.reduced {
        return Err(invalid("row replay disagrees with reduced matrix"));
    }

    let mut rank = 0;
    let mut previous = None;
    let mut zero_row_seen = false;
    let mut contradiction = None;
    for row in 0..rows {
        work.tick(columns)?;
        let pivot = (0..columns).find(|&c| !augmented[row * width + c].is_zero());
        if let Some(column) = pivot {
            if zero_row_seen
                || previous.is_some_and(|p| p >= column)
                || !augmented[row * width + column].is_one()
                || witness.pivots.get(rank) != Some(&column)
            {
                return Err(invalid(
                    "coefficient block is not in reduced row echelon form",
                ));
            }
            for other in 0..rows {
                work.tick(1)?;
                if other != row && !augmented[other * width + column].is_zero() {
                    return Err(invalid("pivot column is not reduced"));
                }
            }
            rank += 1;
            previous = Some(column);
        } else {
            zero_row_seen = true;
            if !augmented[row * width + columns].is_zero() && contradiction.is_none() {
                contradiction = Some(row);
            }
        }
    }
    if witness.pivots.len() != rank {
        return Err(invalid("pivot list does not match coefficient rank"));
    }
    match (&witness.solution, contradiction) {
        (RationalLinearSolution::Inconsistent { row }, Some(expected)) if *row == expected => {}
        (RationalLinearSolution::Inconsistent { .. }, _) | (_, Some(_)) => {
            return Err(invalid("incorrect consistency claim"));
        }
        (RationalLinearSolution::Unique { values }, None) => {
            if rank != columns {
                return Err(invalid("uniqueness requires full column rank"));
            }
            for (row, value) in values.iter().enumerate() {
                work.tick(1)?;
                if value != &augmented[row * width + columns] {
                    return Err(invalid("wrong unique solution"));
                }
            }
        }
        (
            RationalLinearSolution::Affine {
                particular,
                basis,
                free_columns,
            },
            None,
        ) => {
            let nullity = columns - rank;
            if nullity == 0 || basis.len() != nullity {
                return Err(invalid("incomplete or redundant nullspace basis"));
            }
            let mut free_index = 0;
            let mut pivot_index = 0;
            for column in 0..columns {
                work.tick(1)?;
                if witness.pivots.get(pivot_index) == Some(&column) {
                    if particular[column] != augmented[pivot_index * width + columns] {
                        return Err(invalid("incorrect particular solution"));
                    }
                    for (vector, &free) in basis.iter().zip(free_columns) {
                        if free >= columns
                            || !work
                                .add(&vector[column], &augmented[pivot_index * width + free])?
                                .is_zero()
                        {
                            return Err(invalid("incorrect nullspace pivot coordinate"));
                        }
                    }
                    pivot_index += 1;
                } else {
                    if free_columns.get(free_index) != Some(&column)
                        || !particular[column].is_zero()
                    {
                        return Err(invalid("incorrect free-variable coordinates"));
                    }
                    for (index, vector) in basis.iter().enumerate() {
                        work.tick(1)?;
                        if (index == free_index && !vector[column].is_one())
                            || (index != free_index && !vector[column].is_zero())
                        {
                            return Err(invalid("nullspace basis loses a free direction"));
                        }
                    }
                    free_index += 1;
                }
            }
        }
    }
    work.tick(1)?;
    Ok(LinearSystemVerificationReceipt {
        rank,
        nullity: columns - rank,
        consistent: contradiction.is_none(),
        operations: witness.operations.len(),
    })
}
