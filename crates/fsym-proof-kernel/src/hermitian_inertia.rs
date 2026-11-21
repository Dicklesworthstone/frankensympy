//! Exact Gaussian-rational source interpretation and Hermitian-part inertia.
//!
//! For A = B + i C, the realification R(A) = [[B,-C],[C,B]] satisfies
//! R(A) + R(A)^T = R(A + A*). Every eigenvalue of the Hermitian part occurs
//! twice in this real symmetric form. The independent rational block-LDL
//! verifier therefore establishes complete complex inertia after all three
//! counts are checked even. Neither pivot search nor a complex eigensolver
//! participates in this checker.
//!
//! Source interpretation is a bounded exact arithmetic reader, not symbolic
//! simplification. It visits every original child before discarding zeros or
//! zeroth powers, and rejects nonpositive powers of zero. A source-bound
//! entry point reinterprets the original expressions before accepting a
//! generator's coefficient snapshot. Real and imaginary source commitments
//! retain skew-Hermitian parts, even though those do not affect inertia.

use crate::rational_inertia::{
    MAX_RATIONAL_INERTIA_DIMENSION, RationalInertiaError, RationalInertiaReceipt,
    RationalInertiaWitness, verify_rational_inertia,
};
use crate::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, RationalSparseMatrix, Reference, SparseRationalEntry,
    SparseRrefError, check_scalar,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Constant, Expr};
use num_traits::{One, Zero};
use thiserror::Error;

pub const MAX_HERMITIAN_INERTIA_DIMENSION: usize = MAX_RATIONAL_INERTIA_DIMENSION / 2;
pub const MAX_GAUSSIAN_RATIONAL_INPUT_BITS: u64 = 4096;
pub const MAX_GAUSSIAN_SOURCE_NODES: usize = 16_384;
pub const MAX_GAUSSIAN_SOURCE_DEPTH: usize = 64;
pub const MAX_GAUSSIAN_SOURCE_POWER: i64 = 64;

/// The exact original A = real + i imaginary, before taking its Hermitian
/// part. Both matrices have the same square shape and canonical sparse cells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianRationalMatrix {
    pub real: RationalSparseMatrix,
    pub imaginary: RationalSparseMatrix,
}

/// Complete signs of Re(z* A z), equivalently the inertia of (A+A*)/2.
/// For a non-Hermitian source these are not eigenvalue counts of A itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermitianInertiaReceipt {
    positive: usize,
    negative: usize,
    zero: usize,
    real_source_digest: [u8; 32],
    imaginary_source_digest: [u8; 32],
    realification: RationalInertiaReceipt,
}

impl HermitianInertiaReceipt {
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
    pub fn real_source_digest(&self) -> [u8; 32] {
        self.real_source_digest
    }
    pub fn imaginary_source_digest(&self) -> [u8; 32] {
        self.imaginary_source_digest
    }
    pub fn realification(&self) -> &RationalInertiaReceipt {
        &self.realification
    }
}

#[derive(Debug, Error)]
pub enum HermitianInertiaError {
    #[error("Hermitian inertia source must be square, got {0} by {1}")]
    NotSquare(usize, usize),
    #[error("Hermitian inertia requires rational/I arithmetic with bounded integer powers")]
    UnsupportedSource,
    #[error("Hermitian inertia source has a zero base with a nonpositive exponent")]
    UndefinedSource,
    #[error("Hermitian inertia certificate rejected: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Reference(#[from] SparseRrefError),
    #[error(transparent)]
    Rational(#[from] RationalInertiaError),
}

type Result<T> = std::result::Result<T, HermitianInertiaError>;
type Gaussian = (BigRational, BigRational);

fn limit(message: &'static str) -> HermitianInertiaError {
    SparseRrefError::Limit(message).into()
}

fn shape(rows: usize, columns: usize) -> Result<()> {
    if rows != columns {
        return Err(HermitianInertiaError::NotSquare(rows, columns));
    }
    if rows > MAX_HERMITIAN_INERTIA_DIMENSION {
        return Err(limit("Hermitian dimension exceeds 32"));
    }
    Ok(())
}

fn input_coefficient(value: &BigRational) -> Result<()> {
    check_scalar(value)?;
    if value.numer().bits().max(value.denom().bits()) > MAX_GAUSSIAN_RATIONAL_INPUT_BITS {
        return Err(limit(
            "Gaussian-rational input exceeds 4096 coefficient bits",
        ));
    }
    Ok(())
}

fn negate(value: &BigRational, work: &mut Reference<'_, impl BudgetMeter>) -> Result<BigRational> {
    work.tick(value.numer().limb_count() + value.denom().limb_count())?;
    Ok(-work.clone_value(value)?)
}

fn scalar(real: bool, work: &mut Reference<'_, impl BudgetMeter>) -> Result<Gaussian> {
    work.storage(128)?;
    Ok((
        if real {
            BigRational::one()
        } else {
            BigRational::zero()
        },
        BigRational::zero(),
    ))
}

fn multiply(
    left: &Gaussian,
    right: &Gaussian,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Gaussian> {
    let ac = work.multiply(&left.0, &right.0)?;
    let bd = work.multiply(&left.1, &right.1)?;
    let ad = work.multiply(&left.0, &right.1)?;
    let bc = work.multiply(&left.1, &right.0)?;
    let negative_bd = negate(&bd, work)?;
    Ok((work.add(&ac, &negative_bd)?, work.add(&ad, &bc)?))
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
        > MAX_SPARSE_RREF_COEFFICIENT_BITS
        || numerator
            .denom()
            .bits()
            .saturating_add(denominator.numer().bits())
            > MAX_SPARSE_RREF_COEFFICIENT_BITS
    {
        return Err(limit("Gaussian reciprocal intermediate height"));
    }
    work.tick(1)?;
    let value = numerator
        .metered_div(denominator, work)
        .map_err(SparseRrefError::from)?;
    check_scalar(&value)?;
    Ok(value)
}

fn read_scalar(
    expression: &Expr,
    depth: usize,
    visited: &mut usize,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<Gaussian> {
    work.tick(1)?;
    *visited = visited.saturating_add(1);
    if *visited > MAX_GAUSSIAN_SOURCE_NODES || depth > MAX_GAUSSIAN_SOURCE_DEPTH {
        return Err(limit("Gaussian-rational source expression envelope"));
    }
    match expression {
        Expr::Integer(value) => {
            if value.bits() > MAX_GAUSSIAN_RATIONAL_INPUT_BITS {
                return Err(limit(
                    "Gaussian-rational input exceeds 4096 coefficient bits",
                ));
            }
            work.storage(128 + 8 * value.limb_count())?;
            Ok((
                BigRational::from_integer(value.clone()),
                BigRational::zero(),
            ))
        }
        Expr::Rational(value) => {
            input_coefficient(value)?;
            work.storage(64)?;
            Ok((work.clone_value(value)?, BigRational::zero()))
        }
        Expr::Const(Constant::I) => {
            work.storage(128)?;
            Ok((BigRational::zero(), BigRational::one()))
        }
        Expr::Add(children) => {
            if children.len() > MAX_GAUSSIAN_SOURCE_NODES.saturating_sub(*visited) {
                return Err(limit("Gaussian-rational source expression envelope"));
            }
            let mut value = scalar(false, work)?;
            for child in children {
                let next = read_scalar(child, depth + 1, visited, work)?;
                value = (work.add(&value.0, &next.0)?, work.add(&value.1, &next.1)?);
            }
            Ok(value)
        }
        Expr::Mul(children) => {
            if children.len() > MAX_GAUSSIAN_SOURCE_NODES.saturating_sub(*visited) {
                return Err(limit("Gaussian-rational source expression envelope"));
            }
            let mut value = scalar(true, work)?;
            for child in children {
                // Deliberately visit children even after an exact zero factor.
                let next = read_scalar(child, depth + 1, visited, work)?;
                value = multiply(&value, &next, work)?;
            }
            Ok(value)
        }
        Expr::Pow(base, exponent) => {
            let mut base = read_scalar(base, depth + 1, visited, work)?;
            work.tick(1)?;
            *visited = visited.saturating_add(1);
            if *visited > MAX_GAUSSIAN_SOURCE_NODES || depth + 1 > MAX_GAUSSIAN_SOURCE_DEPTH {
                return Err(limit("Gaussian-rational source expression envelope"));
            }
            let Expr::Integer(exponent) = exponent.as_ref() else {
                return Err(HermitianInertiaError::UnsupportedSource);
            };
            let exponent = exponent
                .to_i64()
                .filter(|value| {
                    (-MAX_GAUSSIAN_SOURCE_POWER..=MAX_GAUSSIAN_SOURCE_POWER).contains(value)
                })
                .ok_or_else(|| limit("Gaussian-rational integer power exceeds 64"))?;
            if exponent <= 0 && base.0.is_zero() && base.1.is_zero() {
                return Err(HermitianInertiaError::UndefinedSource);
            }
            if exponent < 0 {
                let real_square = work.multiply(&base.0, &base.0)?;
                let imaginary_square = work.multiply(&base.1, &base.1)?;
                let norm = work.add(&real_square, &imaginary_square)?;
                let imaginary = negate(&base.1, work)?;
                base = (
                    divide(&base.0, &norm, work)?,
                    divide(&imaginary, &norm, work)?,
                );
            }
            let mut remaining = exponent.unsigned_abs();
            let mut value = scalar(true, work)?;
            while remaining != 0 {
                work.tick(1)?;
                if remaining & 1 != 0 {
                    value = multiply(&value, &base, work)?;
                }
                remaining >>= 1;
                if remaining != 0 {
                    base = multiply(&base, &base, work)?;
                }
            }
            Ok(value)
        }
        _ => Err(HermitianInertiaError::UnsupportedSource),
    }
}

fn read_source<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = ((usize, usize), &'a Expr)>,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<GaussianRationalMatrix> {
    work.tick(1)?;
    shape(rows, columns)?;
    let mut real = work.vector(rows * columns)?;
    let mut imaginary = work.vector(rows * columns)?;
    let (mut previous, mut count, mut visited) = (None, 0usize, 0usize);
    for ((row, column), expression) in entries {
        work.tick(1)?;
        count += 1;
        if count > rows * columns
            || row >= rows
            || column >= columns
            || previous.is_some_and(|coordinate| coordinate >= (row, column))
        {
            return Err(HermitianInertiaError::Invalid(
                "source coordinates are not canonical",
            ));
        }
        previous = Some((row, column));
        let (re, im) = read_scalar(expression, 0, &mut visited, work)?;
        input_coefficient(&re)?;
        input_coefficient(&im)?;
        if !re.is_zero() {
            real.push(SparseRationalEntry {
                row,
                column,
                value: re,
            });
        }
        if !im.is_zero() {
            imaginary.push(SparseRationalEntry {
                row,
                column,
                value: im,
            });
        }
    }
    work.tick(1)?;
    Ok(GaussianRationalMatrix {
        real: RationalSparseMatrix {
            rows,
            columns,
            entries: real,
        },
        imaginary: RationalSparseMatrix {
            rows,
            columns,
            entries: imaginary,
        },
    })
}

/// Exact bounded lowering of original borrowed cells. Sparse callers supply
/// only stored coordinates; dense callers may include explicit zeros.
pub fn read_gaussian_rational_matrix<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = ((usize, usize), &'a Expr)>,
    meter: &mut impl BudgetMeter,
) -> Result<GaussianRationalMatrix> {
    read_source(rows, columns, entries, &mut Reference::new(meter))
}

fn check_source(
    source: &GaussianRationalMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    shape(source.real.rows, source.real.columns)?;
    let n = source.real.rows;
    if source.imaginary.rows != n || source.imaginary.columns != n {
        return Err(HermitianInertiaError::Invalid(
            "real and imaginary source shapes differ",
        ));
    }
    for matrix in [&source.real, &source.imaginary] {
        if matrix.entries.len() > n * n {
            return Err(HermitianInertiaError::Invalid(
                "too many Gaussian-rational source cells",
            ));
        }
        work.check_matrix(matrix)?;
        for entry in &matrix.entries {
            work.tick(1)?;
            input_coefficient(&entry.value)?;
        }
    }
    Ok(())
}

fn realification(
    source: &GaussianRationalMatrix,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    let n = source.real.rows;
    let mut entries =
        work.vector(2 * (source.real.entries.len() + source.imaginary.entries.len()))?;
    // Emit canonical rows directly, with no dense temporary or unmetered sort.
    for bottom in [false, true] {
        let (mut real, mut imaginary) = (
            source.real.entries.as_slice(),
            source.imaginary.entries.as_slice(),
        );
        for row in 0..n {
            work.tick(1)?;
            let real_len = real.partition_point(|entry| entry.row == row);
            let imaginary_len = imaginary.partition_point(|entry| entry.row == row);
            let real_row = &real[..real_len];
            let imaginary_row = &imaginary[..imaginary_len];
            let parts = if bottom {
                [imaginary_row, real_row]
            } else {
                [real_row, imaginary_row]
            };
            for (right, part) in parts.into_iter().enumerate() {
                for entry in part {
                    work.tick(1)?;
                    entries.push(SparseRationalEntry {
                        row: row + if bottom { n } else { 0 },
                        column: entry.column + right * n,
                        value: if !bottom && right == 1 {
                            negate(&entry.value, work)?
                        } else {
                            work.clone_value(&entry.value)?
                        },
                    });
                }
            }
            real = &real[real_len..];
            imaginary = &imaginary[imaginary_len..];
        }
    }
    Ok(RationalSparseMatrix {
        rows: 2 * n,
        columns: 2 * n,
        entries,
    })
}

fn verify(
    source: &GaussianRationalMatrix,
    witness: &RationalInertiaWitness,
    work: &mut Reference<'_, impl BudgetMeter>,
) -> Result<HermitianInertiaReceipt> {
    check_source(source, work)?;
    let realified = realification(source, work)?;
    // The outer reference ledger persists across construction, the nested
    // rational checker, source commitments and final publication.
    let realification = verify_rational_inertia(&realified, witness, work)?;
    let counts = [
        realification.positive(),
        realification.negative(),
        realification.zero(),
    ];
    if counts.iter().any(|count| count % 2 != 0)
        || realification.dimension() != 2 * source.real.rows
    {
        return Err(HermitianInertiaError::Invalid(
            "realified inertia is not doubled",
        ));
    }
    let real_source_digest = work.digest(&source.real)?;
    let imaginary_source_digest = work.digest(&source.imaginary)?;
    work.tick(1)?;
    Ok(HermitianInertiaReceipt {
        positive: counts[0] / 2,
        negative: counts[1] / 2,
        zero: counts[2] / 2,
        real_source_digest,
        imaginary_source_digest,
        realification,
    })
}

/// Independently replay a rational congruence against the original exact
/// Gaussian coefficients, not a generator-supplied realification.
pub fn verify_hermitian_inertia(
    source: &GaussianRationalMatrix,
    witness: &RationalInertiaWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<HermitianInertiaReceipt> {
    verify(source, witness, &mut Reference::new(verifier))
}

/// Independently bind the coefficient snapshot to every original expression
/// before checking the complex form. All stages share one verifier envelope.
pub fn verify_hermitian_inertia_source<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = ((usize, usize), &'a Expr)>,
    source: &GaussianRationalMatrix,
    witness: &RationalInertiaWitness,
    verifier: &mut impl BudgetMeter,
) -> Result<HermitianInertiaReceipt> {
    let mut work = Reference::new(verifier);
    check_source(source, &mut work)?;
    let original = read_source(rows, columns, entries, &mut work)?;
    // Both checked snapshots have bounded canonical cells. Charge their exact
    // comparison before equality, including limb reads of all coefficients.
    let comparison = [
        &original.real,
        &original.imaginary,
        &source.real,
        &source.imaginary,
    ]
    .into_iter()
    .flat_map(|matrix| &matrix.entries)
    .map(|entry| 1 + entry.value.numer().limb_count() + entry.value.denom().limb_count())
    .sum();
    work.tick(comparison)?;
    if &original != source {
        return Err(HermitianInertiaError::Invalid(
            "coefficient snapshot differs from original source",
        ));
    }
    verify(source, witness, &mut work)
}
