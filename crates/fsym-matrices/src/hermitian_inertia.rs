//! Complete exact inertia over Q(i), through checked rational realification.
//!
//! The generator lowers bounded original Gaussian-rational expressions and
//! constructs the sparse realification [[Re A,-Im A],[Im A,Re A]]. Existing
//! pivoted rational LDL supplies an untrusted witness. The proof kernel then
//! reinterprets every original cell, independently constructs the real form,
//! checks its full congruence and halves the doubled sign counts.
//!
//! Complex dimension is at most 32, preserving the rational dimension-64
//! envelope. Separate cumulative 20M-step/256MiB phase meters include source
//! interpretation, sparse conversion, LDL and original-source verification.
//! Non-Hermitian matrices classify Re(z* A z), not their eigenvalue signs.

use crate::{Matrix, MatrixError, SparseMatrix};
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigRational, Expr};
use fsym_proof_kernel::hermitian_inertia::{
    GaussianRationalMatrix, HermitianInertiaError, HermitianInertiaReceipt,
    read_gaussian_rational_matrix, verify_hermitian_inertia_source,
};
use fsym_proof_kernel::rational_inertia::{RationalInertiaError, RationalInertiaWitness};
use fsym_proof_kernel::sparse_rref::SparseRrefError;
use std::collections::BTreeMap;

const MAX_STEPS: u64 = 20_000_000;
const MAX_BYTES: u64 = 256 * 1024 * 1024;
type Result<T> = std::result::Result<T, MatrixError>;

/// Borrow the original cells through admission and independent verification.
#[derive(Clone, Copy)]
pub enum HermitianMatrixRef<'a> {
    Dense(&'a Matrix),
    Sparse(&'a SparseMatrix),
}

/// Untrusted exact coefficient snapshot and rational realification witness.
/// Acceptance always receives the expected original matrix separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HermitianInertiaCandidate {
    pub source: GaussianRationalMatrix,
    pub witness: RationalInertiaWitness,
}

/// Complete source-bound Hermitian-part inertia, with independently replayable
/// exact Gaussian coefficients and the full rational congruence witness.
#[derive(Debug)]
pub struct VerifiedHermitianInertia {
    source: GaussianRationalMatrix,
    witness: RationalInertiaWitness,
    receipt: HermitianInertiaReceipt,
}

impl VerifiedHermitianInertia {
    pub fn source(&self) -> &GaussianRationalMatrix {
        &self.source
    }
    pub fn witness(&self) -> &RationalInertiaWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &HermitianInertiaReceipt {
        &self.receipt
    }
}

fn limit(message: &str) -> MatrixError {
    MatrixError::ResourceLimit(format!("Gaussian-rational Hermitian inertia: {message}"))
}

fn stopped(error: MeterError) -> MatrixError {
    match error {
        MeterError::Cancelled => MatrixError::Cancelled,
        MeterError::Budget(error) => MatrixError::ResourceLimit(error.to_string()),
    }
}

fn reference_error(error: SparseRrefError) -> MatrixError {
    match error {
        SparseRrefError::Meter(error) => stopped(error),
        SparseRrefError::Arithmetic(error) => match error {
            RationalArithmeticError::Meter(error) => stopped(error),
            RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
            RationalArithmeticError::InvariantViolation(reason) => {
                MatrixError::ArithmeticInvariantViolation(reason)
            }
            error => limit(&error.to_string()),
        },
        SparseRrefError::Limit(reason) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

fn error(error: HermitianInertiaError) -> MatrixError {
    match error {
        HermitianInertiaError::NotSquare(rows, columns) => MatrixError::NotSquare(rows, columns),
        HermitianInertiaError::UnsupportedSource => MatrixError::UnsupportedCertificateDomain,
        HermitianInertiaError::UndefinedSource => MatrixError::DivisionByZero,
        HermitianInertiaError::Reference(error)
        | HermitianInertiaError::Rational(RationalInertiaError::Reference(error)) => {
            reference_error(error)
        }
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
        }
    }
    fn tick(&mut self, steps: u64) -> Result<()> {
        self.checkpoint().map_err(stopped)?;
        self.charge(Dimension::ComputeSteps, steps.max(1))
            .map_err(stopped)
    }
    fn cell(&mut self, value: &BigRational, negative: bool) -> Result<Expr> {
        let limbs = value.numer().limb_count() + value.denom().limb_count();
        self.tick(limbs + 1)?;
        // Both the temporary BTreeMap and SparseMatrix::new own map nodes.
        // Their cells move without another rational clone.
        self.charge_batch(&[
            (Dimension::MemoryBytes, 512 + 8 * limbs),
            (Dimension::AllocationCount, 3),
        ])
        .map_err(stopped)?;
        Ok(Expr::Rational(if negative {
            -value.clone()
        } else {
            value.clone()
        }))
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let mut next = self.used;
        for &(dimension, amount) in charges {
            let (index, cap) = match dimension {
                Dimension::ComputeSteps => (0, MAX_STEPS),
                Dimension::MemoryBytes => (1, MAX_BYTES),
                Dimension::AllocationCount => (2, MAX_STEPS),
                _ => continue,
            };
            let remaining = cap.saturating_sub(next[index]);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            next[index] += amount;
        }
        self.meter.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }
}

impl HermitianMatrixRef<'_> {
    fn read(self, meter: &mut impl BudgetMeter) -> Result<GaussianRationalMatrix> {
        match self {
            Self::Dense(matrix) => {
                matrix.validate_shape()?;
                read_gaussian_rational_matrix(
                    matrix.rows,
                    matrix.cols,
                    matrix
                        .data
                        .iter()
                        .enumerate()
                        .map(|(index, value)| ((index / matrix.cols, index % matrix.cols), value)),
                    meter,
                )
                .map_err(error)
            }
            Self::Sparse(matrix) => read_gaussian_rational_matrix(
                matrix.rows(),
                matrix.cols(),
                matrix
                    .entries()
                    .iter()
                    .map(|(&coordinate, value)| (coordinate, value)),
                meter,
            )
            .map_err(error),
        }
    }

    fn verify(
        self,
        candidate: &HermitianInertiaCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<HermitianInertiaReceipt> {
        match self {
            Self::Dense(matrix) => {
                matrix.validate_shape()?;
                verify_hermitian_inertia_source(
                    matrix.rows,
                    matrix.cols,
                    matrix
                        .data
                        .iter()
                        .enumerate()
                        .map(|(index, value)| ((index / matrix.cols, index % matrix.cols), value)),
                    &candidate.source,
                    &candidate.witness,
                    meter,
                )
                .map_err(error)
            }
            Self::Sparse(matrix) => verify_hermitian_inertia_source(
                matrix.rows(),
                matrix.cols(),
                matrix
                    .entries()
                    .iter()
                    .map(|(&coordinate, value)| (coordinate, value)),
                &candidate.source,
                &candidate.witness,
                meter,
            )
            .map_err(error),
        }
    }
}

fn generate(
    original: HermitianMatrixRef<'_>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<HermitianInertiaCandidate> {
    work.tick(1)?;
    let source = original.read(work)?;
    let n = source.real.rows;
    let mut entries = BTreeMap::new();
    // This construction is intentionally separate from the verifier's
    // canonical-row reconstruction; a sign/coordinate error cannot certify
    // itself. Every original part is retained, including skew-Hermitian parts.
    for entry in &source.real.entries {
        work.tick(24)?;
        entries.insert((entry.row, entry.column), work.cell(&entry.value, false)?);
        entries.insert(
            (n + entry.row, n + entry.column),
            work.cell(&entry.value, false)?,
        );
    }
    for entry in &source.imaginary.entries {
        work.tick(24)?;
        entries.insert(
            (entry.row, n + entry.column),
            work.cell(&entry.value, true)?,
        );
        entries.insert(
            (n + entry.row, entry.column),
            work.cell(&entry.value, false)?,
        );
    }
    work.tick(entries.len() as u64 * 12 + 1)?;
    let realified = SparseMatrix::new(2 * n, 2 * n, entries)?;
    let candidate = realified.inertia_candidate_with_meter(work)?;
    work.tick(1)?;
    Ok(HermitianInertiaCandidate {
        source,
        witness: candidate.witness,
    })
}

/// Generate an untrusted sparse realification congruence with one cumulative
/// generation envelope, including exact interpretation of all original cells.
pub fn generate_hermitian_inertia_candidate(
    original: HermitianMatrixRef<'_>,
    generator: &mut impl BudgetMeter,
) -> Result<HermitianInertiaCandidate> {
    generate(original, &mut Work::new(generator))
}

/// Check a candidate against the caller's original matrix, never just against
/// the source snapshot carried by the generator.
pub fn verify_hermitian_inertia_candidate(
    original: HermitianMatrixRef<'_>,
    candidate: HermitianInertiaCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedHermitianInertia> {
    let mut work = Work::new(verifier);
    work.tick(1)?;
    let receipt = original.verify(&candidate, &mut work)?;
    work.tick(1)?;
    Ok(VerifiedHermitianInertia {
        source: candidate.source,
        witness: candidate.witness,
        receipt,
    })
}

pub fn metered_hermitian_inertia(
    original: HermitianMatrixRef<'_>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedHermitianInertia> {
    let mut work = Work::new(generator);
    let candidate = generate(original, &mut work)?;
    work.tick(1)?;
    let checked = verify_hermitian_inertia_candidate(original, candidate, verifier)?;
    work.tick(1)?;
    verifier.checkpoint().map_err(stopped)?;
    Ok(checked)
}

impl Matrix {
    /// Bound and interpret original borrowed scalar storage before shell copies.
    pub fn validate_hermitian_inertia_source(&self) -> Result<()> {
        self.validate_hermitian_inertia_source_with_meter(&mut Unbounded)
    }
    pub fn validate_hermitian_inertia_source_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        HermitianMatrixRef::Dense(self).read(&mut Work::new(meter))?;
        Ok(())
    }
    pub fn hermitian_inertia(&self) -> Result<VerifiedHermitianInertia> {
        self.hermitian_inertia_with_meters(&mut Unbounded, &mut Unbounded)
    }
    pub fn hermitian_inertia_with_meters(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedHermitianInertia> {
        metered_hermitian_inertia(HermitianMatrixRef::Dense(self), generator, verifier)
    }
    pub fn hermitian_inertia_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<HermitianInertiaCandidate> {
        generate_hermitian_inertia_candidate(HermitianMatrixRef::Dense(self), generator)
    }
}

impl SparseMatrix {
    pub fn validate_hermitian_inertia_source(&self) -> Result<()> {
        self.validate_hermitian_inertia_source_with_meter(&mut Unbounded)
    }
    pub fn validate_hermitian_inertia_source_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        HermitianMatrixRef::Sparse(self).read(&mut Work::new(meter))?;
        Ok(())
    }
    pub fn hermitian_inertia(&self) -> Result<VerifiedHermitianInertia> {
        self.hermitian_inertia_with_meters(&mut Unbounded, &mut Unbounded)
    }
    pub fn hermitian_inertia_with_meters(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedHermitianInertia> {
        metered_hermitian_inertia(HermitianMatrixRef::Sparse(self), generator, verifier)
    }
    pub fn hermitian_inertia_candidate_with_meter(
        &self,
        generator: &mut impl BudgetMeter,
    ) -> Result<HermitianInertiaCandidate> {
        generate_hermitian_inertia_candidate(HermitianMatrixRef::Sparse(self), generator)
    }
}
