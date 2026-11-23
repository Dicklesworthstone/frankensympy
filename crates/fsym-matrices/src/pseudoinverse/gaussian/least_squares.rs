//! Bounded unique and complete-family Q(i) least squares.
//!
//! R(A*) = R(A)^T, so the independently checked rational full-column-rank
//! solve for R(A), R(B) establishes uniqueness over Q(i). Only then are its
//! left blocks extracted. A distinct direct Q(i) pass re-reads the original
//! A, B and X and checks A* (A X - B) = 0 before publication. Neither a
//! transpose-only normal equation nor the generator's Gram matrix is used
//! to accept a complex result. The unique-solution interface refuses rank
//! deficiency; the separate family interface accepts every minimizer X+PW.
//!
//! Family acceptance reinterprets both original sources and checks an
//! untrusted pseudoinverse with all four Hermitian Penrose identities. It
//! then checks the original normal residual, minimum-norm orthogonality,
//! and a complete orthogonal kernel projector. Residual-only witnesses
//! cannot establish completeness, and a kernel-shifted particular cannot
//! establish minimum norm.

use super::{
    BigRational, BudgetMeter, Expr, GaussianMatrix, MAX_WORK_BITS, Matrix, MatrixError, Operation,
    Result, SparseMatrix, Work, check_bits, check_penrose, limit, propose,
};
use num_traits::{One, Zero};

/// An untrusted least-squares proposal. The pseudoinverse is a completeness
/// witness, not an accepted flag or a substitute for checking A and B.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianLeastSquaresCandidate {
    pub pseudoinverse: Matrix,
    pub minimum_norm: Matrix,
    pub projector: Matrix,
}

/// Every least-squares solution is `minimum_norm + projector * W`, with
/// arbitrary complex W of the minimum-norm matrix's shape. The particular
/// is the unique minimum-Frobenius-norm minimizer. Source snapshots contain
/// the exact Q(i) values admitted from the original expression graphs; the
/// original graphs' domain obligations are checked before these snapshots
/// are normalized. This is an in-process receipt, not a portable certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedGaussianLeastSquaresFamily {
    coefficients: Matrix,
    rhs: Matrix,
    minimum_norm: Matrix,
    projector: Matrix,
}

impl VerifiedGaussianLeastSquaresFamily {
    pub fn coefficients(&self) -> &Matrix {
        &self.coefficients
    }

    pub fn rhs(&self) -> &Matrix {
        &self.rhs
    }

    pub fn minimum_norm(&self) -> &Matrix {
        &self.minimum_norm
    }

    pub fn projector(&self) -> &Matrix {
        &self.projector
    }

    pub fn into_parts(self) -> (Matrix, Matrix) {
        (self.minimum_norm, self.projector)
    }
}

fn extract_solution(
    realified: &SparseMatrix,
    rows: usize,
    cols: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Matrix> {
    work.tick(1)?;
    if realified.rows() != 2 * rows || realified.cols() != 2 * cols {
        return Err(MatrixError::InvalidCertificate(
            "realified least-squares solution shape".into(),
        ));
    }
    let mut solution = GaussianMatrix::zeros(rows, cols, work)?;
    for (&(row, column), expression) in realified.entries() {
        work.tick(1)?;
        if row >= 2 * rows || column >= 2 * cols {
            return Err(MatrixError::InvalidCertificate(
                "realified least-squares solution coordinates".into(),
            ));
        }
        if column >= cols {
            continue;
        }
        let value = match expression {
            Expr::Integer(value) => {
                if value.bits() > MAX_WORK_BITS {
                    return Err(limit("least-squares solution coefficient height"));
                }
                work.tick(value.limb_count() + 1)?;
                work.storage(64 + 8 * value.limb_count())?;
                BigRational::from_integer(value.clone())
            }
            Expr::Rational(value) => {
                check_bits(value, MAX_WORK_BITS)?;
                work.clone_scalar(value)?
            }
            _ => {
                return Err(MatrixError::InvalidCertificate(
                    "realified least-squares solution is not rational".into(),
                ));
            }
        };
        if row < rows {
            solution.entries[row * cols + column].0 = value;
        } else {
            solution.entries[(row - rows) * cols + column].1 = value;
        }
    }
    solution.into_matrix(work)
}

fn check_original_normal_equations(
    matrix: &Matrix,
    rhs: &Matrix,
    solution: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    let a = GaussianMatrix::read(matrix, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    let x = GaussianMatrix::read(solution, work)?;
    check_normal_equations(&a, &b, &x, work)
}

fn check_normal_equations(
    a: &GaussianMatrix,
    b: &GaussianMatrix,
    x: &GaussianMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    if a.rows != b.rows || x.rows != a.cols || x.cols != b.cols {
        return Err(MatrixError::InvalidCertificate(
            "Gaussian least-squares source or solution shape".into(),
        ));
    }
    let mut residual = a.multiply(x, work)?;
    for (value, target) in residual.entries.iter_mut().zip(&b.entries) {
        value.0 = work.rational(&value.0, &target.0, Operation::Subtract)?;
        value.1 = work.rational(&value.1, &target.1, Operation::Subtract)?;
    }
    for variable in 0..a.cols {
        for column in 0..b.cols {
            work.storage(128)?;
            let mut gradient = (BigRational::zero(), BigRational::zero());
            for row in 0..a.rows {
                work.tick(1)?;
                let coefficient = a.get(row, variable);
                let value = residual.get(row, column);
                // (ar - i ai)(rr + i ri) = (ar rr + ai ri)
                //                            + i(ar ri - ai rr).
                let ar_rr = work.rational(&coefficient.0, &value.0, Operation::Multiply)?;
                let ai_ri = work.rational(&coefficient.1, &value.1, Operation::Multiply)?;
                let ar_ri = work.rational(&coefficient.0, &value.1, Operation::Multiply)?;
                let ai_rr = work.rational(&coefficient.1, &value.0, Operation::Multiply)?;
                let product = (
                    work.rational(&ar_rr, &ai_ri, Operation::Add)?,
                    work.rational(&ar_ri, &ai_rr, Operation::Subtract)?,
                );
                gradient = work.add(&gradient, &product)?;
            }
            work.tick(1)?;
            if !gradient.0.is_zero() || !gradient.1.is_zero() {
                return Err(MatrixError::InvalidCertificate(
                    "original-source Hermitian least-squares normal equations failed".into(),
                ));
            }
        }
    }
    work.tick(1)
}

fn identity_minus(
    matrix: &GaussianMatrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianMatrix> {
    if matrix.rows != matrix.cols {
        return Err(MatrixError::InvalidCertificate(
            "kernel projector product is not square".into(),
        ));
    }
    let mut result = GaussianMatrix::zeros(matrix.rows, matrix.cols, work)?;
    for row in 0..matrix.rows {
        for column in 0..matrix.cols {
            work.tick(1)?;
            work.storage(64)?;
            let identity = if row == column {
                BigRational::one()
            } else {
                BigRational::zero()
            };
            let value = matrix.get(row, column);
            result.entries[row * matrix.cols + column] = (
                work.rational(&identity, &value.0, Operation::Subtract)?,
                work.negate(&value.1)?,
            );
        }
    }
    Ok(result)
}

fn require_zero(
    matrix: &GaussianMatrix,
    message: &'static str,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    for value in &matrix.entries {
        work.tick(value.0.numer().limb_count() + value.1.numer().limb_count() + 1)?;
        if !value.0.is_zero() || !value.1.is_zero() {
            return Err(MatrixError::InvalidCertificate(message.into()));
        }
    }
    work.tick(1)
}

fn propose_family(
    matrix: &Matrix,
    rhs: &Matrix,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<GaussianLeastSquaresCandidate> {
    // Inspect both complete originals before zero/rank shortcuts or any
    // multiplication can erase a held reciprocal or unsupported expression.
    let a = GaussianMatrix::read(matrix, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    if a.rows != b.rows {
        return Err(MatrixError::ShapeMismatch(a.rows, a.cols, b.rows, b.cols));
    }
    let pseudoinverse = propose(matrix, work)?;
    let inverse = GaussianMatrix::read(&pseudoinverse, work)?;
    let minimum_norm = inverse.multiply(&b, work)?.into_matrix(work)?;
    let inverse_a = inverse.multiply(&a, work)?;
    let projector = identity_minus(&inverse_a, work)?.into_matrix(work)?;
    work.tick(1)?;
    Ok(GaussianLeastSquaresCandidate {
        pseudoinverse,
        minimum_norm,
        projector,
    })
}

fn verify_family(
    matrix: &Matrix,
    rhs: &Matrix,
    candidate: &GaussianLeastSquaresCandidate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedGaussianLeastSquaresFamily> {
    // This checker never uses the generator's rational RREF, rank, realified
    // block shape or retained source values. All five matrices are read anew.
    let a = GaussianMatrix::read(matrix, work)?;
    let b = GaussianMatrix::read(rhs, work)?;
    let inverse = GaussianMatrix::read(&candidate.pseudoinverse, work)?;
    let x = GaussianMatrix::read(&candidate.minimum_norm, work)?;
    let p = GaussianMatrix::read(&candidate.projector, work)?;
    if a.rows != b.rows
        || x.rows != a.cols
        || x.cols != b.cols
        || p.rows != a.cols
        || p.cols != a.cols
    {
        return Err(MatrixError::InvalidCertificate(
            "Gaussian least-squares family shape".into(),
        ));
    }
    let (_, inverse_a) = check_penrose(&a, &inverse, work)?;
    check_normal_equations(&a, &b, &x, work)?;
    if !p.is_hermitian(work)? {
        return Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is not Hermitian".into(),
        ));
    }
    if !p.multiply(&p, work)?.equal(&p, work)? {
        return Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is not idempotent".into(),
        ));
    }
    require_zero(
        &a.multiply(&p, work)?,
        "least-squares projector leaves the original kernel",
        work,
    )?;
    // AP=0, P=P* and P^2=P do not establish that P spans the entire
    // kernel: even P=0 satisfies them. The independently checked Penrose
    // witness proves that I-A+ A is exactly the complete kernel projector.
    if !p.equal(&identity_minus(&inverse_a, work)?, work)? {
        return Err(MatrixError::InvalidCertificate(
            "least-squares kernel projector is incomplete".into(),
        ));
    }
    require_zero(
        &p.multiply(&x, work)?,
        "least-squares particular is not minimum norm",
        work,
    )?;
    if !inverse.multiply(&b, work)?.equal(&x, work)? {
        return Err(MatrixError::InvalidCertificate(
            "least-squares particular does not match the original RHS".into(),
        ));
    }
    let accepted = VerifiedGaussianLeastSquaresFamily {
        coefficients: a.into_matrix(work)?,
        rhs: b.into_matrix(work)?,
        minimum_norm: x.into_matrix(work)?,
        projector: p.into_matrix(work)?,
    };
    // Include output normalization/allocation and the final publication safe
    // point in the same ledger as source admission and proof replay.
    work.tick(1)?;
    Ok(accepted)
}

impl Matrix {
    /// Propose a bounded exact Q(i) minimum-norm particular and complete
    /// kernel projector. The returned public candidate is untrusted until
    /// checked with [`Self::verify_gaussian_least_squares_candidate`].
    pub fn gaussian_least_squares_candidate_with_meter(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<GaussianLeastSquaresCandidate> {
        propose_family(self, rhs, &mut Work::new(meter))
    }

    /// Independently check the complete complex least-squares family against
    /// the original A and B, without invoking proposal generation. Source,
    /// witness and output are bounded before publication, with cancellation
    /// and all transient arithmetic charged to the supplied meter.
    pub fn verify_gaussian_least_squares_candidate(
        &self,
        rhs: &Self,
        candidate: &GaussianLeastSquaresCandidate,
        meter: &mut impl BudgetMeter,
    ) -> Result<VerifiedGaussianLeastSquaresFamily> {
        verify_family(self, rhs, candidate, &mut Work::new(meter))
    }

    /// Return all exact Q(i) least-squares minimizers as X+PW, including
    /// singular, rectangular, inconsistent and zero-area systems. The unique
    /// minimum-norm minimizer is X. Every axis is at most 32 and each admitted
    /// source/witness/output component has at most 4096 bits. Proposal,
    /// independent original-source verification and publication share one
    /// cumulative caller meter and one 20M-step/256MiB intrinsic envelope.
    /// No protected verifier pool or portable proof family is claimed.
    pub fn metered_gaussian_least_squares_family(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<VerifiedGaussianLeastSquaresFamily> {
        let mut work = Work::new(meter);
        let candidate = propose_family(self, rhs, &mut work)?;
        verify_family(self, rhs, &candidate, &mut work)
    }

    /// Solve exact Q(i) least squares with independently established full
    /// column rank. Both original sources are admitted before generation;
    /// source interpretation, realification, rational generation and rank
    /// replay, direct Hermitian residual checking and publication share one
    /// 20M-step/256MiB ledger. Every axis is at most 32, and source/output
    /// coefficient components have at most 4096 bits. This is a synchronous
    /// reference-checked result, with no protected verifier pool reservation.
    pub fn metered_gaussian_solve_least_squares(
        &self,
        rhs: &Self,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self> {
        let mut work = Work::new(meter);
        let a = GaussianMatrix::read(self, &mut work)?;
        let b = GaussianMatrix::read(rhs, &mut work)?;
        if a.rows != b.rows {
            return Err(MatrixError::ShapeMismatch(a.rows, a.cols, b.rows, b.cols));
        }
        let realified_a = a.realification(&mut work)?;
        let realified_b = b.realification(&mut work)?;
        let checked =
            realified_a.metered_least_squares_with_certificate(&realified_b, &mut work)?;
        // Full column rank is established independently even with no RHS
        // columns. Normal equations alone would not establish uniqueness.
        if checked.rank() != 2 * a.cols {
            return Err(MatrixError::InvalidCertificate(
                "realified least-squares rank does not establish uniqueness".into(),
            ));
        }
        let solution = extract_solution(checked.solution(), a.cols, b.cols, &mut work)?;
        check_original_normal_equations(self, rhs, &solution, &mut work)?;
        work.tick(1)?;
        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_core::Constant;

    #[test]
    fn original_hermitian_residual_rejects_the_transpose_only_answer() {
        let source = Matrix::new(
            2,
            1,
            vec![
                Expr::from_i64(1),
                Expr::from_i64(2) * Expr::Const(Constant::I),
            ],
        )
        .unwrap();
        let rhs = Matrix::new(2, 1, vec![Expr::from_i64(1), Expr::from_i64(0)]).unwrap();
        let wrong = Matrix::new(1, 1, vec![Expr::rational(-1, 3).unwrap()]).unwrap();
        assert!(matches!(
            check_original_normal_equations(&source, &rhs, &wrong, &mut Work::new(&mut Unbounded)),
            Err(MatrixError::InvalidCertificate(_)),
        ));
        let correct = Matrix::new(1, 1, vec![Expr::rational(1, 5).unwrap()]).unwrap();
        check_original_normal_equations(&source, &rhs, &correct, &mut Work::new(&mut Unbounded))
            .unwrap();
        let changed_rhs = Matrix::new(2, 1, vec![Expr::from_i64(2), Expr::from_i64(0)]).unwrap();
        assert!(matches!(
            check_original_normal_equations(
                &source,
                &changed_rhs,
                &correct,
                &mut Work::new(&mut Unbounded),
            ),
            Err(MatrixError::InvalidCertificate(_)),
        ));
    }
}
