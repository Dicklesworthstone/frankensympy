//! Complete integer affine families from a checked Smith decomposition.
//!
//! `A X = B` becomes `D Y = U B` with `X = V Y`. Every nonzero
//! invariant factor must divide the corresponding transformed RHS; every
//! remaining transformed row must vanish. The last columns of unimodular V
//! span the entire integer kernel, including its primitive lattice structure.
//! This is stronger than a rational nullspace or a residual-only check.
//!
//! One generation meter covers both original inputs, Smith generation,
//! transformed RHS, exact divisions, and output. A separate kernel verifier
//! independently checks both sources and the complete family or contradiction.

use super::{
    IntegerMatrix, MAX_INPUT_BITS, Matrix, MatrixError, RationalArithmeticError,
    RationalSparseMatrix, Result, SparseMatrix, SparseRrefError, Work, admit, coefficient,
    generate, limit, snapshot, stopped, verification,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_proof_kernel::integer_linear_system::{
    INTEGER_LINEAR_SCHEMA_VERSION, IntegerLinearError, IntegerLinearOutcome, IntegerLinearReceipt,
    IntegerLinearWitness, verify_integer_linear_system,
};

/// Borrow either built-in native storage format without copying its cells.
/// Both sides may use different formats; all copies occur inside the request's
/// cumulative generation or verification envelope.
#[derive(Debug, Clone, Copy)]
pub enum IntegerMatrixRef<'a> {
    Dense(&'a Matrix),
    Sparse(&'a SparseMatrix),
}

impl<'a> From<&'a Matrix> for IntegerMatrixRef<'a> {
    fn from(matrix: &'a Matrix) -> Self {
        Self::Dense(matrix)
    }
}

impl<'a> From<&'a SparseMatrix> for IntegerMatrixRef<'a> {
    fn from(matrix: &'a SparseMatrix) -> Self {
        Self::Sparse(matrix)
    }
}

impl IntegerMatrixRef<'_> {
    fn shape(self) -> (usize, usize) {
        match self {
            Self::Dense(matrix) => (matrix.rows(), matrix.cols()),
            Self::Sparse(matrix) => (matrix.rows(), matrix.cols()),
        }
    }

    fn admit(self, work: &mut Work<'_, impl BudgetMeter>) -> Result<()> {
        match self {
            Self::Dense(matrix) => {
                matrix.validate_shape()?;
                admit(
                    matrix.rows,
                    matrix.cols,
                    matrix
                        .data
                        .iter()
                        .enumerate()
                        .map(|(index, entry)| ((index / matrix.cols, index % matrix.cols), entry)),
                    work,
                )?;
            }
            Self::Sparse(matrix) => {
                admit(
                    matrix.rows(),
                    matrix.cols(),
                    matrix
                        .entries()
                        .iter()
                        .map(|(&coordinate, value)| (coordinate, value)),
                    work,
                )?;
            }
        }
        Ok(())
    }

    fn snapshot(self, work: &mut Work<'_, impl BudgetMeter>) -> Result<RationalSparseMatrix> {
        match self {
            Self::Dense(matrix) => snapshot(
                matrix.rows,
                matrix.cols,
                matrix
                    .data
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| ((index / matrix.cols, index % matrix.cols), entry)),
                work,
            ),
            Self::Sparse(matrix) => snapshot(
                matrix.rows(),
                matrix.cols(),
                matrix
                    .entries()
                    .iter()
                    .map(|(&coordinate, value)| (coordinate, value)),
                work,
            ),
        }
    }
}

/// An untrusted proposal, retaining both exact original inputs for replay.
/// Acceptance always receives the authoritative A and B separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerLinearCandidate {
    pub source: RationalSparseMatrix,
    pub rhs: RationalSparseMatrix,
    pub witness: IntegerLinearWitness,
}

/// All integer matrices solving the original system, or checked inconsistency.
///
/// A consistent result represents `X = particular + kernel * T` for arbitrary
/// integer T of shape `nullity() x rhs().columns`. The kernel is a complete
/// primitive lattice basis, not merely a full-rank rational nullspace.
#[derive(Debug)]
pub struct VerifiedIntegerLinearSystem {
    source: RationalSparseMatrix,
    rhs: RationalSparseMatrix,
    witness: IntegerLinearWitness,
    receipt: IntegerLinearReceipt,
}

impl VerifiedIntegerLinearSystem {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn rhs(&self) -> &RationalSparseMatrix {
        &self.rhs
    }
    pub fn witness(&self) -> &IntegerLinearWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &IntegerLinearReceipt {
        &self.receipt
    }
    pub fn is_consistent(&self) -> bool {
        self.receipt.is_consistent()
    }
    pub fn is_unique(&self) -> bool {
        self.receipt.is_unique()
    }
    pub fn rank(&self) -> usize {
        self.receipt.rank()
    }
    pub fn nullity(&self) -> usize {
        self.receipt.nullity()
    }
    /// Coordinate of a failed equation in U*B, not the original RHS.
    pub fn contradiction(&self) -> Option<(usize, usize)> {
        self.receipt.contradiction()
    }
    pub fn particular(&self) -> Option<&RationalSparseMatrix> {
        match &self.witness.outcome {
            IntegerLinearOutcome::Solvable { particular, .. } => Some(particular),
            IntegerLinearOutcome::Inconsistent { .. } => None,
        }
    }
    pub fn kernel(&self) -> Option<&RationalSparseMatrix> {
        match &self.witness.outcome {
            IntegerLinearOutcome::Solvable { kernel, .. } => Some(kernel),
            IntegerLinearOutcome::Inconsistent { .. } => None,
        }
    }
    /// Transfer accepted evidence without copying large integer coefficients.
    pub fn into_parts(
        self,
    ) -> (
        RationalSparseMatrix,
        RationalSparseMatrix,
        IntegerLinearWitness,
        IntegerLinearReceipt,
    ) {
        (self.source, self.rhs, self.witness, self.receipt)
    }
}

fn inputs(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(RationalSparseMatrix, RationalSparseMatrix)> {
    // Admit both borrowed originals before any coefficient copies, Smith
    // reduction, zero-RHS shortcut, or no-solution classification.
    source.admit(work)?;
    rhs.admit(work)?;
    let (rows, columns) = source.shape();
    let (rhs_rows, rhs_columns) = rhs.shape();
    if rhs_rows != rows {
        return Err(MatrixError::ShapeMismatch(
            rows,
            columns,
            rhs_rows,
            rhs_columns,
        ));
    }
    Ok((source.snapshot(work)?, rhs.snapshot(work)?))
}

fn multiply(
    left: &IntegerMatrix,
    right: &IntegerMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerMatrix> {
    if left.columns != right.rows {
        return Err(MatrixError::ArithmeticInvariantViolation(
            "integer system multiplication shape",
        ));
    }
    let mut result = IntegerMatrix::zero(left.rows, right.columns, work)?;
    for row in 0..left.rows {
        for inner in 0..left.columns {
            work.tick(1)?;
            let a = left.get(row, inner);
            if a.is_zero() {
                continue;
            }
            for column in 0..right.columns {
                work.tick(1)?;
                let b = right.get(inner, column);
                if b.is_zero() {
                    continue;
                }
                let product = work.multiply(a, b)?;
                // The cooperative integer seam exposes subtraction directly.
                // Negation is bounded and charged before adding the product.
                let negative = work.negate(&product)?;
                let value = work.subtract(result.get(row, column), &negative)?;
                result.values[row * result.columns + column] = value;
            }
        }
    }
    Ok(result)
}

fn outcome(
    normal_form: &fsym_proof_kernel::integer_normal_form::IntegerSmithWitness,
    rhs: &RationalSparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerLinearOutcome> {
    let left = IntegerMatrix::from_source(&normal_form.left, work)?;
    let rhs_values = IntegerMatrix::from_source(rhs, work)?;
    let transformed = multiply(&left, &rhs_values, work)?;
    let rank = normal_form.diagonal.entries.len();
    let variables = normal_form.right.rows;
    let mut coordinates = IntegerMatrix::zero(variables, rhs.columns, work)?;
    for row in 0..transformed.rows {
        for column in 0..transformed.columns {
            work.tick(1)?;
            let value = transformed.get(row, column);
            if row >= rank {
                if !value.is_zero() {
                    return Ok(IntegerLinearOutcome::Inconsistent {
                        row,
                        rhs_column: column,
                    });
                }
            } else {
                let divisor = normal_form.diagonal.entries[row].value.numer();
                let (quotient, remainder) = work.divide(value, divisor)?;
                if !remainder.is_zero() {
                    return Ok(IntegerLinearOutcome::Inconsistent {
                        row,
                        rhs_column: column,
                    });
                }
                coordinates.values[row * rhs.columns + column] = quotient;
            }
        }
    }
    let right = IntegerMatrix::from_source(&normal_form.right, work)?;
    let particular = multiply(&right, &coordinates, work)?.to_sparse(work)?;
    let mut kernel = IntegerMatrix::zero(variables, variables - rank, work)?;
    for row in 0..variables {
        for column in rank..variables {
            kernel.values[row * kernel.columns + column - rank] =
                work.copy(right.get(row, column))?;
        }
    }
    Ok(IntegerLinearOutcome::Solvable {
        particular,
        kernel: kernel.to_sparse(work)?,
    })
}

/// Generate an untrusted complete family or divisibility contradiction.
pub fn generate_integer_linear_candidate(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    generator: &mut impl BudgetMeter,
) -> Result<IntegerLinearCandidate> {
    let mut work = Work::new(generator);
    let (source, rhs) = inputs(source, rhs, &mut work)?;
    let smith = generate(source, &mut work)?;
    let proposed = outcome(&smith.witness, &rhs, &mut work)?;
    work.tick(1)?;
    Ok(IntegerLinearCandidate {
        source: smith.source,
        rhs,
        witness: IntegerLinearWitness {
            schema_version: INTEGER_LINEAR_SCHEMA_VERSION,
            normal_form: smith.witness,
            outcome: proposed,
        },
    })
}

fn same_source(
    original: &RationalSparseMatrix,
    supplied: &RationalSparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    if original.rows != supplied.rows
        || original.columns != supplied.columns
        || original.entries.len() != supplied.entries.len()
    {
        return Err(MatrixError::InvalidCertificate(
            "integer system candidate differs from an original source".into(),
        ));
    }
    for (expected, candidate) in original.entries.iter().zip(&supplied.entries) {
        if !candidate.value.is_integer() || !candidate.value.denom().is_positive() {
            return Err(MatrixError::InvalidCertificate(
                "integer system candidate source contains a noninteger".into(),
            ));
        }
        coefficient(candidate.value.numer(), MAX_INPUT_BITS)?;
        work.tick(expected.value.numer().limb_count().max(1) as usize)?;
        if expected != candidate {
            return Err(MatrixError::InvalidCertificate(
                "integer system candidate differs from an original source".into(),
            ));
        }
    }
    Ok(())
}

fn refused(error: IntegerLinearError) -> MatrixError {
    match error {
        IntegerLinearError::NormalForm(error) => verification(error),
        IntegerLinearError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        IntegerLinearError::Reference(SparseRrefError::Arithmetic(error)) => match error {
            RationalArithmeticError::Meter(error) => stopped(error),
            RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
            RationalArithmeticError::InvariantViolation(reason) => {
                MatrixError::ArithmeticInvariantViolation(reason)
            }
            error => limit(&error.to_string()),
        },
        IntegerLinearError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

/// Bind both originals and independently verify integer completeness.
pub fn verify_integer_linear_candidate(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    candidate: IntegerLinearCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIntegerLinearSystem> {
    let mut work = Work::new(verifier);
    let (source, rhs) = inputs(source, rhs, &mut work)?;
    same_source(&source, &candidate.source, &mut work)?;
    same_source(&rhs, &candidate.rhs, &mut work)?;
    let receipt = verify_integer_linear_system(&source, &rhs, &candidate.witness, &mut work)
        .map_err(refused)?;
    work.tick(1)?;
    Ok(VerifiedIntegerLinearSystem {
        source,
        rhs,
        witness: candidate.witness,
        receipt,
    })
}

/// Solve with separate cumulative generation and independent verifier meters.
/// Mixed dense/sparse sources require no unmetered conversion.
pub fn metered_solve_integer(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIntegerLinearSystem> {
    let candidate = generate_integer_linear_candidate(source, rhs, generator)?;
    generator.checkpoint().map_err(stopped)?;
    let accepted = verify_integer_linear_candidate(source, rhs, candidate, verifier)?;
    generator.checkpoint().map_err(stopped)?;
    verifier.checkpoint().map_err(stopped)?;
    Ok(accepted)
}

impl Matrix {
    /// Complete integer solutions of `self * X = rhs`.
    pub fn solve_integer(&self, rhs: &Matrix) -> Result<VerifiedIntegerLinearSystem> {
        self.metered_solve_integer(rhs, &mut Unbounded, &mut Unbounded)
    }

    pub fn metered_solve_integer(
        &self,
        rhs: &Matrix,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedIntegerLinearSystem> {
        metered_solve_integer(self.into(), rhs.into(), generator, verifier)
    }
}

impl SparseMatrix {
    /// Complete integer solutions without unbounded dense source conversion.
    pub fn solve_integer(&self, rhs: &SparseMatrix) -> Result<VerifiedIntegerLinearSystem> {
        self.metered_solve_integer(rhs, &mut Unbounded, &mut Unbounded)
    }

    pub fn metered_solve_integer(
        &self,
        rhs: &SparseMatrix,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedIntegerLinearSystem> {
        metered_solve_integer(self.into(), rhs.into(), generator, verifier)
    }
}
