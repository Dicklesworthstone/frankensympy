//! Bounded exact rational affine systems, with a separate acceptance boundary.
//!
//! Deterministic Gauss-Jordan generation records every row operation. The
//! proof-kernel checker independently replays that witness and validates the
//! entire solution family, including inconsistency and uniqueness. Generation
//! and verification use distinct caller meters; neither retries with a reset.

use crate::{Matrix, MatrixError, numeric_value};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::BigRational;
use fsym_core::arith::RationalArithmeticError;
use fsym_proof_kernel::linear_system::{
    LINEAR_SYSTEM_SCHEMA_VERSION, LinearRowOperation, LinearSystemVerificationError,
    LinearSystemVerificationReceipt, LinearSystemWitness, MAX_LINEAR_SYSTEM_COEFFICIENT_BITS,
    MAX_LINEAR_SYSTEM_DIMENSION, MAX_LINEAR_SYSTEM_OPERATIONS, RationalLinearSystem,
    verify_linear_system_witness,
};
use num_traits::{One, Zero};

pub use fsym_proof_kernel::linear_system::RationalLinearSolution;

const MAX_WORK: u64 = 20_000_000;
const MAX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;

/// Untrusted generator output. This can be transported or mutated, but only
/// [`verify_affine_solution_candidate`] constructs the accepted wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineSolutionCandidate {
    pub system: RationalLinearSystem,
    pub witness: LinearSystemWitness,
}

/// Exact complete solution set tied to the immutable inputs actually checked.
/// There is no unchecked deserializer or public constructor for this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAffineSolution {
    system: RationalLinearSystem,
    witness: LinearSystemWitness,
    receipt: LinearSystemVerificationReceipt,
}

impl VerifiedAffineSolution {
    pub fn solution(&self) -> &RationalLinearSolution {
        &self.witness.solution
    }
    pub fn rank(&self) -> usize {
        self.receipt.rank()
    }
    pub fn system(&self) -> &RationalLinearSystem {
        &self.system
    }
    pub fn witness(&self) -> &LinearSystemWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &LinearSystemVerificationReceipt {
        &self.receipt
    }
}

fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn arithmetic_error(error: RationalArithmeticError) -> MatrixError {
    match error {
        RationalArithmeticError::Meter(error) => stopped(error),
        RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
        RationalArithmeticError::InvariantViolation(reason) => {
            MatrixError::ArithmeticInvariantViolation(reason)
        }
        error => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn limit(message: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("exact affine system: {message}"))
}

fn check_value(value: &BigRational) -> Result<(), MatrixError> {
    if value.numer().bits().max(value.denom().bits()) > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS {
        return Err(limit("coefficient height exceeds 8192 bits"));
    }
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(MatrixError::ArithmeticInvariantViolation(
            "invalid rational denominator",
        ));
    }
    Ok(())
}

struct Generate<'a, M> {
    meter: &'a mut M,
    steps: u64,
    bytes: u64,
}

impl<M: BudgetMeter> BudgetMeter for Generate<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let (mut steps, mut bytes) = (self.steps, self.bytes);
        for &(dimension, amount) in charges {
            let (counter, maximum) = match dimension {
                Dimension::ComputeSteps => (&mut steps, MAX_WORK),
                Dimension::MemoryBytes => (&mut bytes, MAX_ALLOCATION_BYTES),
                _ => continue,
            };
            let remaining = maximum.saturating_sub(*counter);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *counter += amount;
        }
        self.meter.charge_batch(charges)?;
        self.steps = steps;
        self.bytes = bytes;
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Multiply,
    Divide,
    Subtract,
}

impl<M: BudgetMeter> Generate<'_, M> {
    fn tick(&mut self, amount: usize) -> Result<(), MatrixError> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, amount.max(1) as u64)
            .map_err(stopped)
    }
    fn storage(&mut self, bytes: usize) -> Result<(), MatrixError> {
        self.tick(1)?;
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(stopped)
    }
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, MatrixError> {
        self.storage(capacity.saturating_mul(std::mem::size_of::<T>()))?;
        let mut vector = Vec::new();
        vector
            .try_reserve_exact(capacity)
            .map_err(|_| limit("container allocation refused"))?;
        Ok(vector)
    }
    fn clone_value(&mut self, value: &BigRational) -> Result<BigRational, MatrixError> {
        check_value(value)?;
        self.storage(
            ((value.numer().limb_count() + value.denom().limb_count()).max(1) * 8) as usize,
        )?;
        Ok(value.clone())
    }
    fn rational(
        &mut self,
        a: &BigRational,
        b: &BigRational,
        op: Operation,
    ) -> Result<BigRational, MatrixError> {
        let (an, ad, bn, bd) = (
            a.numer().bits(),
            a.denom().bits(),
            b.numer().bits(),
            b.denom().bits(),
        );
        let (num, den) = match op {
            Operation::Add | Operation::Subtract => (
                an.saturating_add(bd)
                    .max(bn.saturating_add(ad))
                    .saturating_add(1),
                ad.saturating_add(bd),
            ),
            Operation::Multiply => (an.saturating_add(bn), ad.saturating_add(bd)),
            Operation::Divide => (an.saturating_add(bd), ad.saturating_add(bn)),
        };
        if num > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS || den > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS {
            return Err(limit("rational intermediate exceeds coefficient bound"));
        }
        self.tick(1)?;
        let result = match op {
            Operation::Add => a.metered_add(b, self),
            Operation::Subtract => a.metered_sub(b, self),
            Operation::Multiply => a.metered_mul(b, self),
            Operation::Divide => a.metered_div(b, self),
        }
        .map_err(arithmetic_error)?;
        check_value(&result)?;
        Ok(result)
    }
    fn zeros(&mut self, length: usize) -> Result<Vec<BigRational>, MatrixError> {
        let mut values = self.vector(length)?;
        self.storage(length * 16)?;
        values.resize_with(length, BigRational::zero);
        Ok(values)
    }
}

impl Matrix {
    /// Complete exact-rational solution set of `A x = b`, including rectangular
    /// and singular systems. The right-hand side must be a single column.
    ///
    /// At most 128 equations and 128 unknowns are admitted. `0 × n` has every
    /// vector as a solution; an `m × 0` system has the unique empty vector iff
    /// every right-hand side entry is zero, and is otherwise inconsistent.
    pub fn solve_affine(&self, rhs: &Matrix) -> Result<VerifiedAffineSolution, MatrixError> {
        self.solve_affine_with_meters(rhs, &mut Unbounded, &mut Unbounded)
    }

    /// Separate generator and verifier accounting. A caller requiring protected
    /// verification can pass its generator budget and a meter backed by a
    /// reserved verifier lease.
    /// Default hard caps apply to both phases even with unbounded outer meters.
    pub fn solve_affine_with_meters<G: BudgetMeter, V: BudgetMeter>(
        &self,
        rhs: &Matrix,
        generator: &mut G,
        verifier: &mut V,
    ) -> Result<VerifiedAffineSolution, MatrixError> {
        let candidate = self.affine_solution_candidate_with_meter(rhs, generator)?;
        verify_affine_solution_candidate(candidate, verifier)
    }

    /// Generate untrusted row-operation evidence. No acceptance receipt is
    /// created until the independent checker consumes the candidate.
    pub fn affine_solution_candidate_with_meter<M: BudgetMeter>(
        &self,
        rhs: &Matrix,
        meter: &mut M,
    ) -> Result<AffineSolutionCandidate, MatrixError> {
        let mut work = Generate {
            meter,
            steps: 0,
            bytes: 0,
        };
        work.tick(1)?;
        self.validate_shape()?;
        rhs.validate_shape()?;
        let (rows, columns, width) = (self.rows, self.cols, self.cols.saturating_add(1));
        if rows > MAX_LINEAR_SYSTEM_DIMENSION || columns > MAX_LINEAR_SYSTEM_DIMENSION {
            return Err(limit("at most 128 equations and unknowns are supported"));
        }
        if rhs.rows != rows || rhs.cols != 1 {
            return Err(MatrixError::ShapeMismatch(
                rows, columns, rhs.rows, rhs.cols,
            ));
        }
        // Check the entire input before making any arithmetic or coefficient
        // copy, including zero rows and entries irrelevant to pivot discovery.
        for value in self.data.iter().chain(&rhs.data) {
            work.tick(1)?;
            match value {
                fsym_core::Expr::Integer(value) => {
                    if value.bits() > MAX_LINEAR_SYSTEM_COEFFICIENT_BITS {
                        return Err(limit("integer input exceeds coefficient bound"));
                    }
                }
                fsym_core::Expr::Rational(value) => check_value(value)?,
                _ => return Err(MatrixError::UnsupportedCertificateDomain),
            }
        }
        let mut coefficients = work.vector(self.data.len())?;
        let mut rhs_values = work.vector(rows)?;
        for (source, target) in [
            (&self.data, &mut coefficients),
            (&rhs.data, &mut rhs_values),
        ] {
            for entry in source {
                // Account for the owned exact conversion before numeric_value
                // clones an integer or rational payload.
                let limbs = match entry {
                    fsym_core::Expr::Integer(value) => value.limb_count() + 1,
                    fsym_core::Expr::Rational(value) => {
                        value.numer().limb_count() + value.denom().limb_count()
                    }
                    _ => return Err(MatrixError::UnsupportedCertificateDomain),
                };
                work.storage(limbs.max(1) as usize * 8)?;
                target.push(numeric_value(entry).ok_or(MatrixError::UnsupportedCertificateDomain)?);
            }
        }
        let system = RationalLinearSystem {
            rows,
            columns,
            coefficients,
            rhs: rhs_values,
        };
        let mut augmented = work.vector(rows * width)?;
        for row in 0..rows {
            for column in 0..columns {
                augmented.push(work.clone_value(&system.coefficients[row * columns + column])?);
            }
            augmented.push(work.clone_value(&system.rhs[row])?);
        }
        let operation_bound = rows.min(columns) * (rows + 1);
        if operation_bound > MAX_LINEAR_SYSTEM_OPERATIONS {
            return Err(limit("row-operation count"));
        }
        let mut operations = work.vector(operation_bound)?;
        let mut pivots = work.vector(rows.min(columns))?;
        let one = BigRational::one();
        let zero = BigRational::zero();
        for column in 0..columns {
            let row = pivots.len();
            if row == rows {
                break;
            }
            work.tick(rows - row)?;
            let Some(pivot) = (row..rows).find(|&r| !augmented[r * width + column].is_zero())
            else {
                continue;
            };
            if pivot != row {
                work.tick(width)?;
                for c in 0..width {
                    augmented.swap(row * width + c, pivot * width + c);
                }
                operations.push(LinearRowOperation::Swap {
                    left: row,
                    right: pivot,
                });
            }
            if !augmented[row * width + column].is_one() {
                let factor =
                    work.rational(&one, &augmented[row * width + column], Operation::Divide)?;
                for c in 0..width {
                    let index = row * width + c;
                    augmented[index] =
                        work.rational(&augmented[index], &factor, Operation::Multiply)?;
                }
                operations.push(LinearRowOperation::Scale { row, factor });
            }
            for other in 0..rows {
                work.tick(1)?;
                if other == row || augmented[other * width + column].is_zero() {
                    continue;
                }
                let factor = work.rational(
                    &zero,
                    &augmented[other * width + column],
                    Operation::Subtract,
                )?;
                for c in 0..width {
                    let index = other * width + c;
                    let product =
                        work.rational(&augmented[row * width + c], &factor, Operation::Multiply)?;
                    augmented[index] =
                        work.rational(&augmented[index], &product, Operation::Add)?;
                }
                operations.push(LinearRowOperation::Add {
                    target: other,
                    source: row,
                    factor,
                });
            }
            pivots.push(column);
        }
        work.tick(rows * width)?;
        let contradiction =
            (pivots.len()..rows).find(|&row| !augmented[row * width + columns].is_zero());
        let solution = if let Some(row) = contradiction {
            RationalLinearSolution::Inconsistent { row }
        } else {
            let mut particular = work.zeros(columns)?;
            for (row, &column) in pivots.iter().enumerate() {
                particular[column] = work.clone_value(&augmented[row * width + columns])?;
            }
            if pivots.len() == columns {
                RationalLinearSolution::Unique { values: particular }
            } else {
                let nullity = columns - pivots.len();
                let mut free_columns = work.vector(nullity)?;
                let mut basis = work.vector(nullity)?;
                for column in 0..columns {
                    work.tick(pivots.len())?;
                    if pivots.contains(&column) {
                        continue;
                    }
                    free_columns.push(column);
                    let mut vector = work.zeros(columns)?;
                    vector[column] = work.clone_value(&one)?;
                    for (row, &pivot) in pivots.iter().enumerate() {
                        vector[pivot] = work.rational(
                            &zero,
                            &augmented[row * width + column],
                            Operation::Subtract,
                        )?;
                    }
                    basis.push(vector);
                }
                RationalLinearSolution::Affine {
                    particular,
                    basis,
                    free_columns,
                }
            }
        };
        work.tick(1)?;
        Ok(AffineSolutionCandidate {
            system,
            witness: LinearSystemWitness {
                schema_version: LINEAR_SYSTEM_SCHEMA_VERSION,
                operations,
                reduced: augmented,
                pivots,
                solution,
            },
        })
    }
}

/// Independently check a candidate and retain its exact inputs and witness.
/// An unsuccessful call cannot produce `VerifiedAffineSolution`.
pub fn verify_affine_solution_candidate<M: BudgetMeter>(
    candidate: AffineSolutionCandidate,
    verifier: &mut M,
) -> Result<VerifiedAffineSolution, MatrixError> {
    let receipt = verify_linear_system_witness(&candidate.system, &candidate.witness, verifier)
        .map_err(|error| match error {
            LinearSystemVerificationError::Meter(error) => stopped(error),
            LinearSystemVerificationError::Limit(reason) => limit(reason),
            LinearSystemVerificationError::Arithmetic(error) => arithmetic_error(error),
            LinearSystemVerificationError::Invalid(reason) => {
                MatrixError::InvalidCertificate(reason.to_string())
            }
        })?;
    Ok(VerifiedAffineSolution {
        system: candidate.system,
        witness: candidate.witness,
        receipt,
    })
}
