//! Sparse rank-factorization proposals with independent Penrose acceptance.
//!
//! Generation writes `A = C R` using pivot columns and nonzero RREF rows, then
//! solves `Cᵀ C Y = Cᵀ` and `R Rᵀ Z = Y` and proposes `P = Rᵀ Z`.
//! Acceptance does not trust these factors or the generator's rank decisions:
//! the dependency-separated proof kernel checks all four Penrose identities
//! against the exact original source. Minimum-norm solves additionally check
//! `X = P B` in that kernel, including rank-deficient and inconsistent systems.
//!
//! Every matrix stays sparse. Generation, admission, reference replay, retained
//! evidence and result lifting share the structured lane's cumulative Work.
//! This four-identity lane permits twenty million total compute units, with
//! the same 256 MiB cumulative allocation cap; no phase restarts either bound.
//! Portable sources/results have the kernel's 8192-bit coefficient bound;
//! generator arithmetic additionally obeys the structured intermediate bounds.

use super::{
    Arithmetic, MAX_DIMENSION, MAX_ENTRIES, MatrixError, Row, Rows, SparseMatrix, Work,
    from_rational, generate_lu, limit, read_rows, rejected, solve_factors, store, write_rows,
};
use crate::generate_sparse_rref_candidate;
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{BigRational, Expr};
use fsym_proof_kernel::sparse_pseudoinverse::{
    SPARSE_MINIMUM_NORM_SCHEMA_VERSION, SPARSE_PSEUDOINVERSE_SCHEMA_VERSION,
    SparseMinimumNormReceipt, SparseMinimumNormWitness, SparsePseudoinverseError,
    SparsePseudoinverseReceipt, SparsePseudoinverseWitness, verify_sparse_minimum_norm,
    verify_sparse_pseudoinverse,
};
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, RationalSparseMatrix, SparseRationalEntry, SparseRrefError,
};
use num_traits::Zero;
use std::collections::BTreeMap;

const MAX_PSEUDOINVERSE_WORK: u64 = 20_000_000;

/// Portable proposal retained with its original source. No stored flag grants
/// evidence; consuming acceptance binds this source to the receiver's matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsePseudoinverseCandidate {
    pub source: RationalSparseMatrix,
    pub witness: SparsePseudoinverseWitness,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseMinimumNormCandidate {
    pub source: RationalSparseMatrix,
    pub rhs: RationalSparseMatrix,
    pub witness: SparseMinimumNormWitness,
}

/// Original source, portable four-identity proof, and independently accepted P.
/// There is no unchecked constructor or deserializer.
#[derive(Debug)]
pub struct VerifiedSparsePseudoinverse {
    source: RationalSparseMatrix,
    witness: SparsePseudoinverseWitness,
    receipt: SparsePseudoinverseReceipt,
    matrix: SparseMatrix,
}

impl VerifiedSparsePseudoinverse {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn witness(&self) -> &SparsePseudoinverseWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &SparsePseudoinverseReceipt {
        &self.receipt
    }
    pub fn matrix(&self) -> &SparseMatrix {
        &self.matrix
    }
    pub fn into_matrix(self) -> SparseMatrix {
        self.matrix
    }
}

/// The unique minimum-norm least-squares result, its original A/B, and portable
/// independent evidence. This is not the entire affine family of minimizers.
#[derive(Debug)]
pub struct VerifiedSparseMinimumNorm {
    source: RationalSparseMatrix,
    rhs: RationalSparseMatrix,
    witness: SparseMinimumNormWitness,
    receipt: SparseMinimumNormReceipt,
    solution: SparseMatrix,
}

impl VerifiedSparseMinimumNorm {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn rhs(&self) -> &RationalSparseMatrix {
        &self.rhs
    }
    pub fn witness(&self) -> &SparseMinimumNormWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &SparseMinimumNormReceipt {
        &self.receipt
    }
    pub fn solution(&self) -> &SparseMatrix {
        &self.solution
    }
    pub fn into_solution(self) -> SparseMatrix {
        self.solution
    }
}

fn verification_error(error: SparsePseudoinverseError) -> MatrixError {
    match error {
        SparsePseudoinverseError::Invalid(message) => rejected(message),
        SparsePseudoinverseError::Reference(error) => match error {
            SparseRrefError::Invalid(message) => rejected(message),
            SparseRrefError::Limit(message) => limit(message),
            SparseRrefError::Meter(error) => super::meter_error(error),
            SparseRrefError::Arithmetic(error) => super::arithmetic_error(error),
        },
    }
}

fn vector<T, M: BudgetMeter>(count: usize, work: &mut Work<'_, M>) -> Result<Vec<T>, MatrixError> {
    work.reserve((count.saturating_mul(std::mem::size_of::<T>()) as u64).max(1))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| limit("pseudoinverse container allocation"))?;
    Ok(result)
}

fn empty_rows<M: BudgetMeter>(count: usize, work: &mut Work<'_, M>) -> Result<Rows, MatrixError> {
    let mut result = vector(count, work)?;
    result.resize_with(count, Row::new);
    Ok(result)
}

fn portable_scalar(value: &BigRational) -> Result<(), MatrixError> {
    if value.numer().bits().max(value.denom().bits()) > MAX_SPARSE_RREF_COEFFICIENT_BITS {
        return Err(limit("pseudoinverse portable coefficient height"));
    }
    Ok(())
}

fn snapshot<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<RationalSparseMatrix, MatrixError> {
    work.tick(1)?;
    if matrix.rows() > MAX_DIMENSION || matrix.cols() > MAX_DIMENSION {
        return Err(limit("pseudoinverse matrix dimension"));
    }
    if matrix.nnz() > MAX_ENTRIES {
        return Err(limit("pseudoinverse stored entries"));
    }
    // Inspect every original coefficient before cloning, zero or rank shortcuts.
    for (&(row, column), entry) in matrix.entries() {
        work.tick(1)?;
        if row >= matrix.rows() || column >= matrix.cols() {
            return Err(rejected("pseudoinverse source coordinate"));
        }
        match entry {
            Expr::Integer(value) if value.bits() <= MAX_SPARSE_RREF_COEFFICIENT_BITS => {}
            Expr::Integer(_) => return Err(limit("pseudoinverse integer input height")),
            Expr::Rational(value) => portable_scalar(value)?,
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        }
    }
    let mut entries = vector(matrix.nnz(), work)?;
    for (&(row, column), entry) in matrix.entries() {
        work.tick(1)?;
        let value = match entry {
            Expr::Integer(value) => {
                work.tick(value.limb_count().max(1))?;
                work.reserve((value.limb_count().max(1) + 1) * 8)?;
                BigRational::from_integer(value.clone())
            }
            Expr::Rational(value) => work.clone_value(value)?,
            _ => return Err(MatrixError::UnsupportedCertificateDomain),
        };
        if !value.is_zero() {
            entries.push(SparseRationalEntry { row, column, value });
        }
    }
    Ok(RationalSparseMatrix {
        rows: matrix.rows(),
        columns: matrix.cols(),
        entries,
    })
}

fn portable_rows<M: BudgetMeter>(
    rows: Rows,
    columns: usize,
    work: &mut Work<'_, M>,
) -> Result<RationalSparseMatrix, MatrixError> {
    let count: usize = rows.iter().map(Row::len).sum();
    if count > MAX_ENTRIES {
        return Err(limit("pseudoinverse portable fill"));
    }
    let height = rows.len();
    let mut entries = vector(count, work)?;
    for (row, values) in rows.into_iter().enumerate() {
        for (column, value) in values {
            work.tick(1)?;
            portable_scalar(&value)?;
            entries.push(SparseRationalEntry { row, column, value });
        }
    }
    Ok(RationalSparseMatrix {
        rows: height,
        columns,
        entries,
    })
}

fn rows<M: BudgetMeter>(
    matrix: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<Rows, MatrixError> {
    let mut result = empty_rows(matrix.rows, work)?;
    let mut count = 0;
    for entry in &matrix.entries {
        let value = work.clone_value(&entry.value)?;
        store(
            &mut result[entry.row],
            entry.column,
            value,
            &mut count,
            work,
        )?;
    }
    Ok(result)
}

fn transpose<M: BudgetMeter>(
    source: &[Row],
    columns: usize,
    work: &mut Work<'_, M>,
) -> Result<Rows, MatrixError> {
    let mut result = empty_rows(columns, work)?;
    let mut count = 0;
    for (row, values) in source.iter().enumerate() {
        work.tick(1)?;
        for (&column, value) in values {
            let value = work.clone_value(value)?;
            store(&mut result[column], row, value, &mut count, work)?;
        }
    }
    Ok(result)
}

fn product<M: BudgetMeter>(
    left: &[Row],
    right: &[Row],
    work: &mut Work<'_, M>,
) -> Result<Rows, MatrixError> {
    let mut result = empty_rows(left.len(), work)?;
    let mut count = 0;
    for (row, values) in left.iter().enumerate() {
        work.tick(1)?;
        for (&inner, coefficient) in values {
            for (&column, value) in &right[inner] {
                let term = work.arithmetic(coefficient, value, Arithmetic::Multiply)?;
                let sum = match result[row].get(&column) {
                    Some(prior) => work.arithmetic(prior, &term, Arithmetic::Add)?,
                    None => term,
                };
                store(&mut result[row], column, sum, &mut count, work)?;
            }
        }
    }
    Ok(result)
}

fn generate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparsePseudoinverseCandidate, MatrixError> {
    let reduction = generate_sparse_rref_candidate(matrix, work)?;
    let source = reduction.source;
    let rank = reduction.witness.pivots.len();
    let pseudoinverse = if rank == 0 {
        portable_rows(empty_rows(source.columns, work)?, source.rows, work)?
    } else {
        let mut pivot_index = vector(source.columns, work)?;
        pivot_index.resize(source.columns, None);
        for (index, &pivot) in reduction.witness.pivots.iter().enumerate() {
            work.tick(1)?;
            pivot_index[pivot] = Some(index);
        }
        let mut c = empty_rows(source.rows, work)?;
        let mut count = 0;
        for entry in &source.entries {
            work.tick(1)?;
            if let Some(column) = pivot_index[entry.column] {
                let value = work.clone_value(&entry.value)?;
                store(&mut c[entry.row], column, value, &mut count, work)?;
            }
        }
        let mut r = empty_rows(rank, work)?;
        count = 0;
        for entry in reduction.witness.reduced.entries {
            work.tick(1)?;
            if entry.row < rank {
                store(
                    &mut r[entry.row],
                    entry.column,
                    entry.value,
                    &mut count,
                    work,
                )?;
            }
        }
        let ct = transpose(&c, rank, work)?;
        let rt = transpose(&r, source.columns, work)?;
        let ct_c = write_rows(product(&ct, &c, work)?, rank, work)?;
        let r_rt = write_rows(product(&r, &rt, work)?, rank, work)?;
        let ct = write_rows(ct, source.rows, work)?;
        let c_factor = generate_lu(&ct_c, work)?;
        let c_plus = solve_factors(&ct, &c_factor, work)?;
        let r_factor = generate_lu(&r_rt, work)?;
        let z = solve_factors(&c_plus, &r_factor, work)?;
        let z = read_rows(&z, work)?;
        portable_rows(product(&rt, &z, work)?, source.rows, work)?
    };
    work.tick(1)?;
    Ok(SparsePseudoinverseCandidate {
        source,
        witness: SparsePseudoinverseWitness {
            schema_version: SPARSE_PSEUDOINVERSE_SCHEMA_VERSION,
            pseudoinverse,
        },
    })
}

fn bind<M: BudgetMeter>(
    matrix: &SparseMatrix,
    claimed: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<(), MatrixError> {
    let expected = snapshot(matrix, work)?;
    if (expected.rows, expected.columns, expected.entries.len())
        != (claimed.rows, claimed.columns, claimed.entries.len())
    {
        return Err(rejected("pseudoinverse original source mismatch"));
    }
    for (left, right) in expected.entries.iter().zip(&claimed.entries) {
        portable_scalar(&right.value)?;
        work.tick(
            left.value.numer().limb_count()
                + left.value.denom().limb_count()
                + right.value.numer().limb_count()
                + right.value.denom().limb_count(),
        )?;
        if left != right {
            return Err(rejected("pseudoinverse original source mismatch"));
        }
    }
    Ok(())
}

fn lift<M: BudgetMeter>(
    matrix: &RationalSparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseMatrix, MatrixError> {
    work.reserve((matrix.entries.len() as u64).saturating_mul(256))?;
    let mut entries = BTreeMap::new();
    for entry in &matrix.entries {
        let value = work.clone_value(&entry.value)?;
        entries.insert((entry.row, entry.column), from_rational(value));
    }
    SparseMatrix::new(matrix.rows, matrix.columns, entries)
}

fn accept<M: BudgetMeter>(
    matrix: &SparseMatrix,
    candidate: SparsePseudoinverseCandidate,
    work: &mut Work<'_, M>,
) -> Result<VerifiedSparsePseudoinverse, MatrixError> {
    bind(matrix, &candidate.source, work)?;
    let receipt = verify_sparse_pseudoinverse(&candidate.source, &candidate.witness, work)
        .map_err(verification_error)?;
    let matrix = lift(&candidate.witness.pseudoinverse, work)?;
    work.tick(1)?;
    Ok(VerifiedSparsePseudoinverse {
        source: candidate.source,
        witness: candidate.witness,
        receipt,
        matrix,
    })
}

fn generate_solve<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    work: &mut Work<'_, M>,
) -> Result<SparseMinimumNormCandidate, MatrixError> {
    if matrix.rows() != rhs.rows() {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows(),
            matrix.cols(),
            rhs.rows(),
            rhs.cols(),
        ));
    }
    let rhs = snapshot(rhs, work)?;
    let candidate = generate(matrix, work)?;
    let p = rows(&candidate.witness.pseudoinverse, work)?;
    let b = rows(&rhs, work)?;
    let solution = portable_rows(product(&p, &b, work)?, rhs.columns, work)?;
    work.tick(1)?;
    Ok(SparseMinimumNormCandidate {
        source: candidate.source,
        rhs,
        witness: SparseMinimumNormWitness {
            schema_version: SPARSE_MINIMUM_NORM_SCHEMA_VERSION,
            pseudoinverse: candidate.witness,
            solution,
        },
    })
}

fn accept_solve<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    candidate: SparseMinimumNormCandidate,
    work: &mut Work<'_, M>,
) -> Result<VerifiedSparseMinimumNorm, MatrixError> {
    bind(matrix, &candidate.source, work)?;
    bind(rhs, &candidate.rhs, work)?;
    let receipt =
        verify_sparse_minimum_norm(&candidate.source, &candidate.rhs, &candidate.witness, work)
            .map_err(verification_error)?;
    let solution = lift(&candidate.witness.solution, work)?;
    work.tick(1)?;
    Ok(VerifiedSparseMinimumNorm {
        source: candidate.source,
        rhs: candidate.rhs,
        witness: candidate.witness,
        receipt,
        solution,
    })
}

pub fn generate_sparse_pseudoinverse_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    meter: &mut M,
) -> Result<SparsePseudoinverseCandidate, MatrixError> {
    generate(
        matrix,
        &mut Work::with_step_limit(meter, MAX_PSEUDOINVERSE_WORK),
    )
}

pub fn verify_sparse_pseudoinverse_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    candidate: SparsePseudoinverseCandidate,
    meter: &mut M,
) -> Result<VerifiedSparsePseudoinverse, MatrixError> {
    accept(
        matrix,
        candidate,
        &mut Work::with_step_limit(meter, MAX_PSEUDOINVERSE_WORK),
    )
}

pub fn generate_sparse_minimum_norm_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    meter: &mut M,
) -> Result<SparseMinimumNormCandidate, MatrixError> {
    generate_solve(
        matrix,
        rhs,
        &mut Work::with_step_limit(meter, MAX_PSEUDOINVERSE_WORK),
    )
}

pub fn verify_sparse_minimum_norm_candidate<M: BudgetMeter>(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    candidate: SparseMinimumNormCandidate,
    meter: &mut M,
) -> Result<VerifiedSparseMinimumNorm, MatrixError> {
    accept_solve(
        matrix,
        rhs,
        candidate,
        &mut Work::with_step_limit(meter, MAX_PSEUDOINVERSE_WORK),
    )
}

impl SparseMatrix {
    pub fn pseudoinverse(&self) -> Result<Self, MatrixError> {
        self.metered_pseudoinverse(&mut Unbounded)
    }

    pub fn metered_pseudoinverse<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<Self, MatrixError> {
        self.metered_pseudoinverse_with_certificate(meter)
            .map(VerifiedSparsePseudoinverse::into_matrix)
    }

    pub fn pseudoinverse_with_certificate(
        &self,
    ) -> Result<VerifiedSparsePseudoinverse, MatrixError> {
        self.metered_pseudoinverse_with_certificate(&mut Unbounded)
    }

    pub fn metered_pseudoinverse_with_certificate<M: BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<VerifiedSparsePseudoinverse, MatrixError> {
        let mut work = Work::with_step_limit(meter, MAX_PSEUDOINVERSE_WORK);
        let candidate = generate(self, &mut work)?;
        accept(self, candidate, &mut work)
    }

    /// The unique minimum-norm minimizer of `||self * X - rhs||²` over Q.
    /// Rank deficiency and inconsistent original equations are supported.
    pub fn minimum_norm_least_squares(&self, rhs: &Self) -> Result<Self, MatrixError> {
        self.metered_minimum_norm_least_squares(rhs, &mut Unbounded)
    }

    pub fn metered_minimum_norm_least_squares<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<Self, MatrixError> {
        self.metered_minimum_norm_with_certificate(rhs, meter)
            .map(VerifiedSparseMinimumNorm::into_solution)
    }

    pub fn minimum_norm_with_certificate(
        &self,
        rhs: &Self,
    ) -> Result<VerifiedSparseMinimumNorm, MatrixError> {
        self.metered_minimum_norm_with_certificate(rhs, &mut Unbounded)
    }

    pub fn metered_minimum_norm_with_certificate<M: BudgetMeter>(
        &self,
        rhs: &Self,
        meter: &mut M,
    ) -> Result<VerifiedSparseMinimumNorm, MatrixError> {
        let mut work = Work::with_step_limit(meter, MAX_PSEUDOINVERSE_WORK);
        let candidate = generate_solve(self, rhs, &mut work)?;
        accept_solve(self, rhs, candidate, &mut work)
    }
}
