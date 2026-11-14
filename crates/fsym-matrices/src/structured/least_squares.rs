//! Unique rational least squares with sparse proposals and source-bound checks.

use super::{
    Arithmetic, MatrixError, Row, Rows, SparseMatrix, Work, generate_lu, read_rows, rejected,
    solve_factors, store, write_rows,
};
use crate::{
    SparseRrefCandidate, VerifiedSparseRref, generate_sparse_rref_candidate,
    verify_sparse_rref_candidate,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::BigRational;
use num_traits::Zero;

/// Untrusted least-squares solution and a rank witness for the original matrix.
///
/// The normal matrix used during generation is deliberately absent: acceptance
/// checks the original-source normal equations, and independently replays the
/// rank witness against the actual coefficient matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseLeastSquaresCandidate {
    pub solution: SparseMatrix,
    pub rank_witness: SparseRrefCandidate,
}

/// Accepted unique minimizer of `||A X - B||²` over exact rational matrices.
///
/// Full column rank is established by the independent portable RREF checker.
/// The original-source normal equations are checked by sparse exact arithmetic
/// in this crate; this combined result is not a portable least-squares proof.
/// There is no public unchecked constructor or deserializer.
#[derive(Debug)]
pub struct VerifiedSparseLeastSquares {
    solution: SparseMatrix,
    rhs: SparseMatrix,
    rank_evidence: VerifiedSparseRref,
}

impl VerifiedSparseLeastSquares {
    pub fn solution(&self) -> &SparseMatrix {
        &self.solution
    }

    pub fn rhs(&self) -> &SparseMatrix {
        &self.rhs
    }

    /// Includes the exact original coefficient source and accepted rank witness.
    pub fn rank_evidence(&self) -> &VerifiedSparseRref {
        &self.rank_evidence
    }

    pub fn rank(&self) -> usize {
        self.rank_evidence.rank()
    }

    pub fn into_solution(self) -> SparseMatrix {
        self.solution
    }
}

fn input_shapes(matrix: &SparseMatrix, rhs: &SparseMatrix) -> Result<(), MatrixError> {
    if matrix.rows() != rhs.rows() {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows(),
            matrix.cols(),
            rhs.rows(),
            rhs.cols(),
        ));
    }
    Ok(())
}

fn empty_rows<M: BudgetMeter>(count: usize, work: &mut Work<'_, M>) -> Result<Rows, MatrixError> {
    work.reserve((count as u64).saturating_mul(32))?;
    Ok(vec![Row::new(); count])
}

fn add_product<M: BudgetMeter>(
    target: &mut Row,
    column: usize,
    left: &BigRational,
    right: &BigRational,
    count: &mut usize,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    let product = work.arithmetic(left, right, Arithmetic::Multiply)?;
    let sum = match target.get(&column) {
        Some(prior) => work.arithmetic(prior, &product, Arithmetic::Add)?,
        None => product,
    };
    store(target, column, sum, count, work)
}

fn generate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseLeastSquaresCandidate, MatrixError> {
    input_shapes(matrix, rhs)?;
    // Admit every source coefficient before rank, empty-RHS or zero shortcuts.
    let coefficients = read_rows(matrix, work)?;
    let right_hand_side = read_rows(rhs, work)?;
    if matrix.rows() < matrix.cols() {
        return Err(MatrixError::SingularMatrix);
    }
    // Nested row-reduction accounting is charged through this same Work, so
    // it cannot reset the aggregate work or allocation limits between phases.
    let rank_witness = generate_sparse_rref_candidate(matrix, work)?;
    if rank_witness.witness.pivots.len() != matrix.cols() {
        return Err(MatrixError::SingularMatrix);
    }
    let mut gram = empty_rows(matrix.cols(), work)?;
    let mut projected_rhs = empty_rows(matrix.cols(), work)?;
    let mut gram_count = 0;
    let mut rhs_count = 0;
    for (row, values) in coefficients.iter().enumerate() {
        work.tick(1)?;
        for (&column, value) in values {
            for (&other_column, other_value) in values {
                add_product(
                    &mut gram[column],
                    other_column,
                    value,
                    other_value,
                    &mut gram_count,
                    work,
                )?;
            }
            for (&rhs_column, rhs_value) in &right_hand_side[row] {
                add_product(
                    &mut projected_rhs[column],
                    rhs_column,
                    value,
                    rhs_value,
                    &mut rhs_count,
                    work,
                )?;
            }
        }
    }
    let gram = write_rows(gram, matrix.cols(), work)?;
    let projected_rhs = write_rows(projected_rhs, rhs.cols(), work)?;
    let factorization = generate_lu(&gram, work)?;
    let solution = solve_factors(&projected_rhs, &factorization, work)?;
    work.tick(1)?;
    Ok(SparseLeastSquaresCandidate {
        solution,
        rank_witness,
    })
}

fn accept<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    candidate: SparseLeastSquaresCandidate,
    work: &mut Work<'_, M>,
) -> Result<VerifiedSparseLeastSquares, MatrixError> {
    input_shapes(matrix, rhs)?;
    if candidate.solution.rows() != matrix.cols() || candidate.solution.cols() != rhs.cols() {
        return Err(rejected(
            "sparse least-squares solution has the wrong shape",
        ));
    }
    let coefficients = read_rows(matrix, work)?;
    let right_hand_side = read_rows(rhs, work)?;
    let solution = read_rows(&candidate.solution, work)?;
    let rank_evidence = verify_sparse_rref_candidate(matrix, candidate.rank_witness, work)?;
    if rank_evidence.rank() != matrix.cols() {
        return Err(rejected(
            "sparse least-squares rank witness does not establish uniqueness",
        ));
    }

    // Recompute A^T (A X - B) directly from the original admitted rows.
    // Only one residual row is retained; neither A X nor a dense matrix is
    // allocated. This code does not consult a normal matrix or solve factors.
    let mut gradient = empty_rows(matrix.cols(), work)?;
    let mut gradient_count = 0;
    for (row, values) in coefficients.iter().enumerate() {
        work.tick(1)?;
        let mut residual = Row::new();
        let mut residual_count = 0;
        for (&column, value) in values {
            for (&rhs_column, solution_value) in &solution[column] {
                add_product(
                    &mut residual,
                    rhs_column,
                    value,
                    solution_value,
                    &mut residual_count,
                    work,
                )?;
            }
        }
        for (&rhs_column, value) in &right_hand_side[row] {
            let zero = BigRational::zero();
            let prior = residual.get(&rhs_column).unwrap_or(&zero);
            let difference = work.arithmetic(prior, value, Arithmetic::Subtract)?;
            store(
                &mut residual,
                rhs_column,
                difference,
                &mut residual_count,
                work,
            )?;
        }
        for (&column, value) in values {
            for (&rhs_column, residual_value) in &residual {
                add_product(
                    &mut gradient[column],
                    rhs_column,
                    value,
                    residual_value,
                    &mut gradient_count,
                    work,
                )?;
            }
        }
    }
    for row in gradient {
        work.tick(1)?;
        if !row.is_empty() {
            return Err(rejected(
                "original-source sparse least-squares normal equations failed",
            ));
        }
    }
    let rhs = write_rows(right_hand_side, rhs.cols(), work)?;
    work.tick(1)?;
    Ok(VerifiedSparseLeastSquares {
        solution: candidate.solution,
        rhs,
        rank_evidence,
    })
}

/// Propose a sparse unique rational least-squares result without accepting it.
///
/// All inputs, Gram fill, arithmetic intermediates and logical allocations are
/// bounded. Rank-witness generation additionally uses the RREF coefficient
/// limit of 8192 bits; other rational arithmetic uses the 16384-bit sparse
/// elimination limit. Nonunique and underdetermined systems are refused.
pub fn generate_sparse_least_squares_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    meter: &mut M,
) -> Result<SparseLeastSquaresCandidate, MatrixError> {
    generate(matrix, rhs, &mut Work::new(meter))
}

/// Independently bind and accept a candidate against both actual input matrices.
///
/// The rank witness must replay against the original `matrix`, and its solution
/// must satisfy `matrix^T * (matrix * solution - rhs) = 0` exactly. The candidate
/// is consumed to avoid an unmetered copy of its portable rank witness.
pub fn verify_sparse_least_squares_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    candidate: SparseLeastSquaresCandidate,
    meter: &mut M,
) -> Result<VerifiedSparseLeastSquares, MatrixError> {
    accept(matrix, rhs, candidate, &mut Work::new(meter))
}

impl SparseMatrix {
    /// Unique exact rational least squares, with sparse inputs and outputs.
    ///
    /// This minimizes the sum of squared residuals independently for every RHS
    /// column. Full column rank is required even for a zero or empty RHS.
    pub fn solve_least_squares(&self, rhs: &Self) -> Result<Self, MatrixError> {
        self.metered_solve_least_squares(rhs, &mut Unbounded)
    }

    /// Sparse generation and independent acceptance under one cumulative meter.
    pub fn metered_solve_least_squares<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, MatrixError> {
        Ok(self
            .metered_least_squares_with_certificate(rhs, meter)?
            .into_solution())
    }

    /// Return the accepted solution, original RHS and independent rank evidence.
    pub fn least_squares_with_certificate(
        &self,
        rhs: &Self,
    ) -> Result<VerifiedSparseLeastSquares, MatrixError> {
        self.metered_least_squares_with_certificate(rhs, &mut Unbounded)
    }

    /// One ledger covers source admission, generation, checking and publication.
    /// This synchronous API does not reserve a protected verifier budget; callers
    /// needing separate reserves can use the proposal and acceptance functions.
    pub fn metered_least_squares_with_certificate<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<VerifiedSparseLeastSquares, MatrixError> {
        let mut work = Work::new(meter);
        let candidate = generate(self, rhs, &mut work)?;
        accept(self, rhs, candidate, &mut work)
    }
}
