//! Unique bounded Q(i) least squares through exact realification.
//!
//! R(A*) = R(A)^T, so the independently checked rational full-column-rank
//! solve for R(A), R(B) establishes uniqueness over Q(i). Only then are its
//! left blocks extracted. A distinct direct Q(i) pass re-reads the original
//! A, B and X and checks A* (A X - B) = 0 before publication. Neither a
//! transpose-only normal equation nor the generator's Gram matrix is used
//! to accept a complex result. Rank-deficient systems remain unsupported by
//! this unique-solution interface; complete affine families are unchanged.

use super::{
    BigRational, BudgetMeter, Expr, GaussianMatrix, MAX_WORK_BITS, Matrix, MatrixError, Operation,
    Result, SparseMatrix, Work, check_bits, limit,
};
use num_traits::Zero;

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
    if a.rows != b.rows || x.rows != a.cols || x.cols != b.cols {
        return Err(MatrixError::InvalidCertificate(
            "Gaussian least-squares source or solution shape".into(),
        ));
    }
    let mut residual = a.multiply(&x, work)?;
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

impl Matrix {
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
