//! Complete column Hermite forms with integer lattice witnesses.
//!
//! Column operations alone transform A into `[0 | H]`. Rows are processed
//! from bottom to top, placing the next pivot in the rightmost unused column.
//! Euclidean division clears the remaining entries of that row, then positive
//! pivots reduce the entries to their right to canonical nonnegative residues.
//! Both the integral column transformation and its integral inverse are kept.
//!
//! The independent proof kernel checks canonical structure, `A V = [0 | H]`
//! and both inverse identities. This proves equality of the full integer
//! column lattices, including rank-deficient and rectangular inputs; matching
//! rational column spaces or a one-way product identity would be insufficient.
//!
//! Shared integer normal-form arithmetic bounds dimensions at 64, input
//! coefficients at 4096 bits and intermediate coefficients at 8192 bits. Each
//! of the separate generation and verification phases retains one cumulative
//! 20M-step/256MiB envelope through publication, with cooperative safe points.

use super::{
    IntegerMatrix, MAX_INPUT_BITS, Matrix, MatrixError, RationalArithmeticError,
    RationalSparseMatrix, Result, SparseMatrix, SparseRationalEntry, SparseRrefError, Work,
    coefficient, limit, snapshot, stopped,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_proof_kernel::integer_hermite::{
    INTEGER_HERMITE_SCHEMA_VERSION, IntegerHermiteError, IntegerHermiteReceipt,
    IntegerHermiteWitness, verify_integer_hermite,
};

/// An untrusted form and its source. Verification separately receives the
/// authoritative original, so a substituted source cannot gain acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermiteNormalFormCandidate {
    pub source: RationalSparseMatrix,
    pub witness: IntegerHermiteWitness,
}

/// Immutable, independently checked column-lattice normal form of the source.
#[derive(Debug)]
pub struct VerifiedHermiteNormalForm {
    source: RationalSparseMatrix,
    witness: IntegerHermiteWitness,
    receipt: IntegerHermiteReceipt,
}

impl VerifiedHermiteNormalForm {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    /// H has every original row and exactly `rank()` independent columns.
    pub fn normal_form(&self) -> &RationalSparseMatrix {
        &self.witness.normal_form
    }
    /// Integral unimodular V satisfying `source * V = [0 | H]`.
    pub fn right(&self) -> &RationalSparseMatrix {
        &self.witness.right
    }
    pub fn right_inverse(&self) -> &RationalSparseMatrix {
        &self.witness.right_inverse
    }
    pub fn rank(&self) -> usize {
        self.receipt.rank()
    }
    pub fn witness(&self) -> &IntegerHermiteWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &IntegerHermiteReceipt {
        &self.receipt
    }
    /// Move the accepted source, witness and receipt without coefficient copies.
    pub fn into_parts(
        self,
    ) -> (
        RationalSparseMatrix,
        IntegerHermiteWitness,
        IntegerHermiteReceipt,
    ) {
        (self.source, self.witness, self.receipt)
    }
}

fn verification(error: IntegerHermiteError) -> MatrixError {
    match error {
        IntegerHermiteError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        IntegerHermiteError::Reference(SparseRrefError::Arithmetic(error)) => match error {
            RationalArithmeticError::Meter(error) => stopped(error),
            RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
            RationalArithmeticError::InvariantViolation(reason) => {
                MatrixError::ArithmeticInvariantViolation(reason)
            }
            error => limit(&error.to_string()),
        },
        IntegerHermiteError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

struct Decomposition {
    transformed: IntegerMatrix,
    right: IntegerMatrix,
    inverse: IntegerMatrix,
}

impl Decomposition {
    fn swap_columns(
        &mut self,
        left: usize,
        right: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.transformed.swap_columns(left, right, work)?;
        self.right.swap_columns(left, right, work)?;
        self.inverse.swap_rows(left, right, work)
    }

    fn subtract_column(
        &mut self,
        target: usize,
        source: usize,
        multiple: &fsym_core::BigInt,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.transformed
            .subtract_column(target, source, multiple, work)?;
        self.right.subtract_column(target, source, multiple, work)?;
        let opposite = work.negate(multiple)?;
        // V' = V E implies V'^-1 = E^-1 V^-1. The inverse therefore
        // updates the opposite (source) row with the opposite multiplier.
        self.inverse.subtract_row(source, target, &opposite, work)
    }

    fn negate_column(
        &mut self,
        column: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<()> {
        self.transformed.negate_column(column, work)?;
        self.right.negate_column(column, work)?;
        self.inverse.negate_row(column, work)
    }

    fn reduce(&mut self, work: &mut Work<'_, impl BudgetMeter>) -> Result<usize> {
        let mut first = self.transformed.columns;
        for row in (0..self.transformed.rows).rev() {
            work.tick(1)?;
            if first == 0 {
                break;
            }
            let pivot = first - 1;
            let mut selected = None;
            let mut bits = u64::MAX;
            for column in 0..first {
                work.tick(1)?;
                let value = self.transformed.get(row, column);
                if !value.is_zero() && value.bits() < bits {
                    selected = Some(column);
                    bits = value.bits();
                }
            }
            let Some(column) = selected else { continue };
            self.swap_columns(column, pivot, work)?;

            // Each nonzero remainder strictly reduces the pivot's absolute
            // value. Already processed lower rows stay zero in these columns.
            loop {
                work.tick(1)?;
                let mut reduced = false;
                for column in 0..pivot {
                    work.tick(1)?;
                    if self.transformed.get(row, column).is_zero() {
                        continue;
                    }
                    let (quotient, remainder) = work.divide(
                        self.transformed.get(row, column),
                        self.transformed.get(row, pivot),
                    )?;
                    self.subtract_column(column, pivot, &quotient, work)?;
                    if !remainder.is_zero() {
                        self.swap_columns(column, pivot, work)?;
                        reduced = true;
                        break;
                    }
                }
                if !reduced {
                    break;
                }
            }
            if self.transformed.get(row, pivot).is_negative() {
                self.negate_column(pivot, work)?;
            }
            for column in first..self.transformed.columns {
                work.tick(1)?;
                let (mut quotient, remainder) = work.divide(
                    self.transformed.get(row, column),
                    self.transformed.get(row, pivot),
                )?;
                // Integer division truncates toward zero; HNF requires the
                // Euclidean representative, including for negative entries.
                if remainder.is_negative() {
                    let one = work.scalar(1)?;
                    quotient = work.subtract(&quotient, &one)?;
                }
                self.subtract_column(column, pivot, &quotient, work)?;
            }
            first = pivot;
        }
        Ok(first)
    }

    fn form(
        &self,
        first: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<RationalSparseMatrix> {
        let columns = self.transformed.columns - first;
        let rows = self.transformed.rows;
        work.tick(rows * columns)?;
        let mut count = 0;
        for row in 0..rows {
            for column in first..self.transformed.columns {
                count += usize::from(!self.transformed.get(row, column).is_zero());
            }
        }
        let mut entries = work.vector(count)?;
        for row in 0..rows {
            for column in first..self.transformed.columns {
                work.tick(1)?;
                let value = self.transformed.get(row, column);
                if !value.is_zero() {
                    entries.push(SparseRationalEntry {
                        row,
                        column: column - first,
                        value: work.rational(value)?,
                    });
                }
            }
        }
        Ok(RationalSparseMatrix {
            rows,
            columns,
            entries,
        })
    }
}

fn generate(
    source: RationalSparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<HermiteNormalFormCandidate> {
    let mut decomposition = Decomposition {
        transformed: IntegerMatrix::from_source(&source, work)?,
        right: IntegerMatrix::identity(source.columns, work)?,
        inverse: IntegerMatrix::identity(source.columns, work)?,
    };
    let first = decomposition.reduce(work)?;
    let witness = IntegerHermiteWitness {
        schema_version: INTEGER_HERMITE_SCHEMA_VERSION,
        normal_form: decomposition.form(first, work)?,
        right: decomposition.right.to_sparse(work)?,
        right_inverse: decomposition.inverse.to_sparse(work)?,
    };
    work.tick(1)?;
    Ok(HermiteNormalFormCandidate { source, witness })
}

fn accept(
    original: RationalSparseMatrix,
    candidate: HermiteNormalFormCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedHermiteNormalForm> {
    if original.rows != candidate.source.rows
        || original.columns != candidate.source.columns
        || original.entries.len() != candidate.source.entries.len()
    {
        return Err(MatrixError::InvalidCertificate(
            "Hermite source differs from the original matrix".into(),
        ));
    }
    for (expected, supplied) in original.entries.iter().zip(&candidate.source.entries) {
        if !supplied.value.is_integer() || !supplied.value.denom().is_positive() {
            return Err(MatrixError::InvalidCertificate(
                "Hermite candidate source contains a noninteger".into(),
            ));
        }
        coefficient(supplied.value.numer(), MAX_INPUT_BITS)?;
        work.tick(expected.value.numer().limb_count().max(1) as usize)?;
        if expected != supplied {
            return Err(MatrixError::InvalidCertificate(
                "Hermite source differs from the original matrix".into(),
            ));
        }
    }
    let receipt =
        verify_integer_hermite(&original, &candidate.witness, work).map_err(verification)?;
    work.tick(1)?;
    Ok(VerifiedHermiteNormalForm {
        source: original,
        witness: candidate.witness,
        receipt,
    })
}

fn dense_source(
    source: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    source.validate_shape()?;
    snapshot(
        source.rows,
        source.cols,
        source
            .data
            .iter()
            .enumerate()
            .map(|(i, value)| ((i / source.cols, i % source.cols), value)),
        work,
    )
}

fn sparse_source(
    source: &SparseMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    snapshot(
        source.rows(),
        source.cols(),
        source
            .entries()
            .iter()
            .map(|(&coordinate, value)| (coordinate, value)),
        work,
    )
}

/// Produce an untrusted complete column-lattice proposal.
pub fn generate_hermite_normal_form_candidate(
    source: &Matrix,
    generator: &mut impl BudgetMeter,
) -> Result<HermiteNormalFormCandidate> {
    source.hermite_normal_form_candidate_with_meter(generator)
}

/// Bind a proposal to the original matrix and independently verify canonical
/// HNF structure and integer-unimodular column equivalence.
pub fn verify_hermite_normal_form_candidate(
    source: &Matrix,
    candidate: HermiteNormalFormCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedHermiteNormalForm> {
    let mut work = Work::new(verifier);
    let original = dense_source(source, &mut work)?;
    accept(original, candidate, &mut work)
}

impl Matrix {
    pub fn hermite_normal_form(&self) -> Result<VerifiedHermiteNormalForm> {
        self.metered_hermite_normal_form(&mut Unbounded, &mut Unbounded)
    }

    pub fn metered_hermite_normal_form(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedHermiteNormalForm> {
        let mut generation = Work::new(generator);
        let source = dense_source(self, &mut generation)?;
        let candidate = generate(source, &mut generation)?;
        generation.tick(1)?;
        let mut verification = Work::new(verifier);
        let original = dense_source(self, &mut verification)?;
        let accepted = accept(original, candidate, &mut verification)?;
        generation.tick(1)?;
        verification.tick(1)?;
        Ok(accepted)
    }

    pub fn hermite_normal_form_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<HermiteNormalFormCandidate> {
        let mut work = Work::new(generator);
        let source = dense_source(self, &mut work)?;
        generate(source, &mut work)
    }
}

impl SparseMatrix {
    /// Sparse entries are admitted before bounded dense elimination storage.
    pub fn hermite_normal_form(&self) -> Result<VerifiedHermiteNormalForm> {
        self.metered_hermite_normal_form(&mut Unbounded, &mut Unbounded)
    }

    pub fn metered_hermite_normal_form(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedHermiteNormalForm> {
        let mut generation = Work::new(generator);
        let source = sparse_source(self, &mut generation)?;
        let candidate = generate(source, &mut generation)?;
        generation.tick(1)?;
        let mut verification = Work::new(verifier);
        let original = sparse_source(self, &mut verification)?;
        let accepted = accept(original, candidate, &mut verification)?;
        generation.tick(1)?;
        verification.tick(1)?;
        Ok(accepted)
    }
}
