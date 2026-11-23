//! Exact Q(i) linear systems, with an independently checked left inverse.
//!
//! A proposed L and X are accepted only after re-reading the original A and
//! B and checking L A = I, X = L B and A X = B in exact Gaussian arithmetic.
//! The first identity establishes uniqueness, including when B has no columns.
//! The last distinguishes an exact solution from a least-squares projection.
//! The generator's realification, rank and normal equations have no authority
//! at this acceptance boundary. All work shares the existing bounded ledger.

use super::{
    BudgetMeter, Constant, Expr, GaussianMatrix, Matrix, MatrixError, Result, Work, limit, propose,
};
use fsym_proof_kernel::hermitian_inertia::{MAX_GAUSSIAN_SOURCE_DEPTH, MAX_GAUSSIAN_SOURCE_NODES};
use num_traits::{One, Zero};

/// Untrusted source-relative proposal for a unique exact linear solution.
/// The left inverse establishes uniqueness independently of the generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianLinearSystemCandidate {
    pub left_inverse: Matrix,
    pub solution: Matrix,
}

#[derive(Clone, Copy)]
enum ExactCheck {
    Accepted,
    NoLeftInverse,
    Inconsistent,
}

fn is_identity(matrix: &GaussianMatrix, work: &mut Work<'_, impl BudgetMeter>) -> Result<bool> {
    if matrix.rows != matrix.cols {
        return Ok(false);
    }
    for row in 0..matrix.rows {
        for column in 0..matrix.cols {
            let value = matrix.get(row, column);
            work.tick(value.0.numer().limb_count() + value.1.numer().limb_count() + 1)?;
            if !value.1.is_zero()
                || if row == column {
                    !value.0.is_one()
                } else {
                    !value.0.is_zero()
                }
            {
                return Ok(false);
            }
        }
    }
    work.tick(1)?;
    Ok(true)
}

fn generate(
    matrix: &Matrix,
    rhs: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianLinearSystemCandidate> {
    // Admit both originals before singular/empty shortcuts or arithmetic can
    // erase a held undefined child. These are not accepted source receipts.
    let a = GaussianMatrix::read(matrix, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    if a.rows != b.rows {
        return Err(MatrixError::ShapeMismatch(a.rows, a.cols, b.rows, b.cols));
    }
    let left_inverse = propose(matrix, work)?;
    let inverse = GaussianMatrix::read(&left_inverse, work)?;
    let solution = inverse.multiply(&b, work)?.into_matrix(work)?;
    work.tick(1)?;
    Ok(GaussianLinearSystemCandidate {
        left_inverse,
        solution,
    })
}

fn check(
    matrix: &Matrix,
    rhs: &Matrix,
    candidate: &GaussianLinearSystemCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ExactCheck> {
    // Deliberately independent of proposal generation, sparse matrix
    // products, RREF, symbolic simplification and cached source reductions.
    let a = GaussianMatrix::read(matrix, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    let left = GaussianMatrix::read(&candidate.left_inverse, work)?;
    let x = GaussianMatrix::read(&candidate.solution, work)?;
    if a.rows != b.rows
        || left.rows != a.cols
        || left.cols != a.rows
        || x.rows != a.cols
        || x.cols != b.cols
    {
        return Err(MatrixError::InvalidCertificate(
            "Gaussian exact linear-system source or witness shape".into(),
        ));
    }
    if !is_identity(&left.multiply(&a, work)?, work)? {
        return Ok(ExactCheck::NoLeftInverse);
    }
    // Once L A=I, every possible solution must equal L B. Check this before
    // distinguishing inconsistency from a merely malformed proposed X.
    if !left.multiply(&b, work)?.equal(&x, work)? {
        return Err(MatrixError::InvalidCertificate(
            "Gaussian exact solution does not equal the original left-inverse RHS product".into(),
        ));
    }
    if !a.multiply(&x, work)?.equal(&b, work)? {
        return Ok(ExactCheck::Inconsistent);
    }
    work.tick(1)?;
    Ok(ExactCheck::Accepted)
}

fn gaussian_source_syntax<'a>(
    roots: impl IntoIterator<Item = &'a Expr>,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    let mut visited = 0usize;
    let mut pending = work.vector::<(&Expr, usize)>(1)?;
    for original in roots {
        pending.push((original, 0));
        while let Some((expression, depth)) = pending.pop() {
            work.tick(1)?;
            visited += 1;
            if visited > MAX_GAUSSIAN_SOURCE_NODES || depth > MAX_GAUSSIAN_SOURCE_DEPTH {
                return Err(limit("Gaussian-rational source syntax envelope"));
            }
            match expression {
                Expr::Integer(_) | Expr::Rational(_) | Expr::Const(Constant::I) => {}
                Expr::Add(children) | Expr::Mul(children) => {
                    if children.len() > MAX_GAUSSIAN_SOURCE_NODES - visited - pending.len() {
                        return Err(limit("Gaussian-rational source syntax envelope"));
                    }
                    work.storage((children.len() * std::mem::size_of::<(&Expr, usize)>()) as u64)?;
                    pending
                        .try_reserve(children.len())
                        .map_err(|_| limit("Gaussian source syntax allocation refused"))?;
                    pending.extend(children.iter().map(|child| (child, depth + 1)));
                }
                Expr::Pow(base, exponent) if matches!(exponent.as_ref(), Expr::Integer(_)) => {
                    if pending.len() + 2 > MAX_GAUSSIAN_SOURCE_NODES - visited {
                        return Err(limit("Gaussian-rational source syntax envelope"));
                    }
                    work.storage((2 * std::mem::size_of::<(&Expr, usize)>()) as u64)?;
                    pending
                        .try_reserve(2)
                        .map_err(|_| limit("Gaussian source syntax allocation refused"))?;
                    pending.push((base, depth + 1));
                    pending.push((exponent, depth + 1));
                }
                _ => return Ok(false),
            }
        }
    }
    work.tick(1)?;
    Ok(true)
}

impl super::SparseMatrix {
    /// Inspect stored syntax without densifying or imposing Gaussian axis
    /// limits on symbolic sparse matrices. True selects a candidate lane;
    /// independent source admission still checks dimensions and all domains.
    pub fn has_gaussian_rational_syntax_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<bool> {
        gaussian_source_syntax(self.entries().values(), &mut Work::new(meter))
    }
}

impl Matrix {
    /// Select closed rational/I arithmetic before an inverse algorithm runs.
    /// This only inspects bounded syntax; it never grants mathematical
    /// admission or erases domain obligations such as a held zero reciprocal.
    pub fn has_gaussian_rational_syntax_with_meter(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<bool> {
        self.validate_shape()?;
        gaussian_source_syntax(self.data(), &mut Work::new(meter))
    }

    /// Generate an untrusted bounded Q(i) linear-system proposal. Only the
    /// separate reference checker can establish existence and uniqueness.
    pub fn gaussian_linear_system_candidate_with_meter(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianLinearSystemCandidate> {
        generate(self, rhs, &mut Work::new(meter))
    }

    /// Check L A=I, X=L B and A X=B against both original expression graphs.
    /// This proves the unique exact solution; a valid residual alone does not.
    /// It is an in-process reference check, not a portable proof certificate.
    pub fn verify_gaussian_linear_system_candidate(
        &self,
        rhs: &Self,
        candidate: &GaussianLinearSystemCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<()> {
        match check(self, rhs, candidate, &mut Work::new(meter))? {
            ExactCheck::Accepted => Ok(()),
            ExactCheck::NoLeftInverse => Err(MatrixError::InvalidCertificate(
                "Gaussian left-inverse witness does not establish uniqueness".into(),
            )),
            ExactCheck::Inconsistent => Err(MatrixError::InvalidCertificate(
                "Gaussian exact solution has a nonzero original-source residual".into(),
            )),
        }
    }

    /// Solve a unique exact Q(i) system, including consistent tall systems.
    /// Every axis is bounded by 32 and source/output components by 4096 bits.
    /// Admission, proposal, independent checking and publication share one
    /// 20M-step/256MiB ledger. Nonunique systems refuse; a least-squares
    /// projection can never be returned for an inconsistent exact request.
    pub fn metered_gaussian_solve(&self, rhs: &Self, meter: &mut impl BudgetMeter) -> Result<Self> {
        let mut work = Work::new(meter);
        let candidate = generate(self, rhs, &mut work)?;
        match check(self, rhs, &candidate, &mut work)? {
            ExactCheck::Accepted => {
                work.tick(1)?;
                Ok(candidate.solution)
            }
            // Failure to produce a left inverse refuses the request; it is
            // never published as an accepted singularity certificate.
            ExactCheck::NoLeftInverse => Err(MatrixError::SingularMatrix),
            ExactCheck::Inconsistent => Err(MatrixError::InconsistentLinearSystem),
        }
    }

    /// Compute an exact Q(i) inverse and independently check both products
    /// against the original source, under the same cumulative envelope.
    pub fn metered_gaussian_inverse(&self, meter: &mut impl BudgetMeter) -> Result<Self> {
        let mut work = Work::new(meter);
        let a = GaussianMatrix::read(self, &mut work)?;
        if a.rows != a.cols {
            return Err(MatrixError::NotSquare(a.rows, a.cols));
        }
        let inverse = propose(self, &mut work)?;
        let left = GaussianMatrix::read(&inverse, &mut work)?;
        // Read A again at acceptance rather than trusting the generator's
        // reduced/realified source or its rank classification.
        let original = GaussianMatrix::read(self, &mut work)?;
        if !is_identity(&left.multiply(&original, &mut work)?, &mut work)?
            || !is_identity(&original.multiply(&left, &mut work)?, &mut work)?
        {
            return Err(MatrixError::SingularMatrix);
        }
        work.tick(1)?;
        Ok(inverse)
    }
}
