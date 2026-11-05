//! Checked bounded matrix exponentials from an accepted fundamental matrix.
//!
//! The source is a square matrix of literal rational entries, dimension at
//! most eight. One independently checked general ODE receipt supplies exp(A*t).
//! Its accepted block basis is evaluated exactly at t=1. No eigenvector or
//! Jordan-form heuristic is used, and no approximate rank decision is made.
//! Sparse admission visits stored entries only; bounded result fill is charged
//! before materializing coordinates. Spectral factors must be rational linear
//! or quadratic; higher-degree irreducible factors refuse.

use crate::{Matrix, MatrixError, SparseMatrix};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_solvers::ode::system::general::{GeneralOdeSystemError, metered_general_ode_system};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use thiserror::Error;

const MAX_DIMENSION: usize = 8;
const MAX_INPUT_BITS: u64 = 1024;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum MatrixExponentialError {
    #[error("matrix exponential requires square input, got {0}x{1}")]
    NotSquare(usize, usize),
    #[error("matrix exponential supports literal exact rational entries only")]
    UnsupportedDomain,
    #[error("matrix exponential resource limit: {0}")]
    ResourceLimit(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Ode(GeneralOdeSystemError),
    #[error(transparent)]
    Matrix(#[from] MatrixError),
}

impl MatrixExponentialError {
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::UnsupportedDomain)
            || matches!(self, Self::Ode(error) if error.is_unsupported())
    }
}

impl From<GeneralOdeSystemError> for MatrixExponentialError {
    fn from(error: GeneralOdeSystemError) -> Self {
        match error {
            GeneralOdeSystemError::Meter(error) => Self::Meter(error),
            error => Self::Ode(error),
        }
    }
}

type Result<T> = std::result::Result<T, MatrixExponentialError>;

struct Capped<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let mut next = self.used;
        for &(dimension, amount) in charges {
            let (index, limit) = match dimension {
                Dimension::ComputeSteps => (0, MAX_COMPUTE),
                Dimension::MemoryBytes => (1, MAX_MEMORY),
                Dimension::AllocationCount => (2, MAX_COMPUTE),
                _ => continue,
            };
            let remaining = limit.saturating_sub(next[index]);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            next[index] += amount;
        }
        self.inner.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

fn charge(meter: &mut impl BudgetMeter, steps: usize, bytes: usize) -> Result<()> {
    meter.checkpoint()?;
    meter.charge_batch(&[
        (Dimension::ComputeSteps, steps.max(1) as u64),
        (Dimension::MemoryBytes, bytes.max(1) as u64),
        (Dimension::AllocationCount, steps.max(1) as u64),
    ])?;
    Ok(())
}

/// Borrowed dimensions and every literal cell precede clones and fill.
fn admit<'a>(
    rows: usize,
    columns: usize,
    entries: impl Iterator<Item = &'a Expr>,
    meter: &mut impl BudgetMeter,
) -> Result<usize> {
    meter.checkpoint()?;
    if rows != columns {
        return Err(MatrixExponentialError::NotSquare(rows, columns));
    }
    if rows > MAX_DIMENSION {
        return Err(MatrixExponentialError::ResourceLimit(
            "dimension exceeds eight",
        ));
    }
    for (index, value) in entries.enumerate() {
        if index >= rows * rows {
            return Err(MatrixExponentialError::ResourceLimit("source entry count"));
        }
        match value {
            Expr::Integer(value) if value.bits() <= MAX_INPUT_BITS => {}
            Expr::Rational(value)
                if value.denom().is_positive()
                    && value.numer().bits().max(value.denom().bits()) <= MAX_INPUT_BITS => {}
            Expr::Integer(_) | Expr::Rational(_) => {
                return Err(MatrixExponentialError::ResourceLimit(
                    "input rational coefficient exceeds 1024 bits",
                ));
            }
            _ => return Err(MatrixExponentialError::UnsupportedDomain),
        }
    }
    charge(meter, rows * rows + 1, 512 * (rows * rows + rows + 1))?;
    Ok(rows)
}

fn rational(value: &Expr) -> BigRational {
    match value {
        Expr::Integer(value) => BigRational::from_integer(value.clone()),
        Expr::Rational(value) => value.clone(),
        _ => unreachable!("literal source was admitted"),
    }
}

fn evaluate(
    coefficients: &[Vec<BigRational>],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Vec<Vec<Expr>>> {
    let n = coefficients.len();
    if n == 0 {
        charge(verifier, 1, 64)?;
        return Ok(Vec::new());
    }
    charge(generator, n * 2 + 1, 512 * (n * 2 + 1))?;
    let forcing = vec![vec![BigRational::zero()]; n];
    let initial = vec![Expr::from_i64(0); n];
    // No source symbols exist in the admitted rational matrix.
    let time = Symbol::new("_fsym_matrix_exponential");
    let checked = metered_general_ode_system(
        coefficients,
        &time,
        &BigRational::zero(),
        &forcing,
        &initial,
        generator,
        verifier,
    )?;
    Ok(checked.fundamental_at(&BigRational::one(), verifier)?)
}

impl Matrix {
    /// Exact rational matrix exponential with checked ODE evidence.
    pub fn exp(&self) -> Result<Self> {
        self.metered_exp(&mut Unbounded, &mut Unbounded)
    }

    /// The same cumulative meters cover admission, the complete fundamental
    /// matrix, exact evaluation, result allocation and final publication.
    pub fn metered_exp(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<Self> {
        let mut generation = Capped {
            inner: generator,
            used: [0; 3],
        };
        let mut verification = Capped {
            inner: verifier,
            used: [0; 3],
        };
        let n = admit(
            self.rows(),
            self.cols(),
            self.data().iter(),
            &mut generation,
        )?;
        let coefficients: Vec<Vec<_>> = self
            .data()
            .chunks(n.max(1))
            .map(|row| row.iter().map(rational).collect())
            .collect();
        let values = evaluate(&coefficients, &mut generation, &mut verification)?;
        charge(&mut verification, n * n + 1, 256 * (n * n + 1))?;
        let result = Matrix::new(n, n, values.into_iter().flatten().collect())?;
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(result)
    }
}

impl SparseMatrix {
    /// Exact rational exponential; result fill remains explicitly bounded.
    pub fn exp(&self) -> Result<Self> {
        self.metered_exp(&mut Unbounded, &mut Unbounded)
    }

    pub fn metered_exp(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<Self> {
        let mut generation = Capped {
            inner: generator,
            used: [0; 3],
        };
        let mut verification = Capped {
            inner: verifier,
            used: [0; 3],
        };
        let n = admit(
            self.rows(),
            self.cols(),
            self.entries().values(),
            &mut generation,
        )?;
        let mut coefficients = vec![vec![BigRational::zero(); n]; n];
        for (&(row, column), value) in self.entries() {
            if row >= n || column >= n {
                return Err(MatrixExponentialError::ResourceLimit(
                    "invalid stored coordinate",
                ));
            }
            coefficients[row][column] = rational(value);
        }
        let values = evaluate(&coefficients, &mut generation, &mut verification)?;
        charge(&mut verification, n * n * 2 + 1, 1024 * (n * n + 1))?;
        let mut entries = BTreeMap::new();
        for (row, values) in values.into_iter().enumerate() {
            for (column, value) in values.into_iter().enumerate() {
                if !value.is_zero() {
                    entries.insert((row, column), value);
                }
            }
        }
        let result = SparseMatrix::new(n, n, entries)?;
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(result)
    }
}
