//! Exact-rational Moore-Penrose pseudoinverses and a reference identity checker.

use super::{MAX_MATRIX_MULTIPLICATION_OPS, Matrix, MatrixError, numeric_value};
use fsym_core::BigRational;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

impl Matrix {
    /// Computes the Moore-Penrose pseudoinverse of an exact-rational matrix.
    ///
    /// Rank factorization handles singular, rectangular, and zero matrices.
    /// If `A = C F`, the pivot columns `C` have full column rank and the
    /// nonzero RREF rows `F` have full row rank. Consequently
    /// `A+ = F^T (F F^T)^-1 (C^T C)^-1 C^T`.
    ///
    /// The existing matrix storage, multiplication, and elimination limits
    /// apply to each sub-operation, rather than to one aggregate budget.
    /// All four Penrose identities are checked by a separate exact rational
    /// reference calculation before a result is returned. Symbolic and
    /// non-rational inputs retain an explicit unsupported-domain error.
    pub fn pseudoinverse(&self) -> Result<Self, MatrixError> {
        self.validate_certificate_domain()?;
        if self.data.iter().all(|entry| entry.is_zero()) {
            let inverse = Matrix::zeros(self.cols, self.rows)?;
            verify_pseudoinverse_certificate(
                self,
                &PseudoinverseCertificate {
                    inverse: inverse.clone(),
                },
            )?;
            return Ok(inverse);
        }

        let (reduced, pivots) = self.rref()?;
        let rank = pivots.len();
        let mut column_data = Vec::with_capacity(Self::checked_element_count(self.rows, rank)?);
        for row in 0..self.rows {
            for &column in &pivots {
                column_data.push(self.get(row, column)?.clone());
            }
        }
        let columns = Matrix::new(self.rows, rank, column_data)?;
        let factor_entries = Self::checked_element_count(rank, self.cols)?;
        let rows = Matrix::new(rank, self.cols, reduced.data[..factor_entries].to_vec())?;

        let ct = columns.transpose();
        let ft = rows.transpose();
        let left_inverse = ct.matmul(&columns)?.solve(&ct)?;
        // Solve rather than explicitly invert the two positive-definite
        // rational Gram matrices. Transposition gives F^T (F F^T)^-1.
        let right_inverse = rows.matmul(&ft)?.solve(&rows)?.transpose();
        let inverse = right_inverse.matmul(&left_inverse)?;
        verify_pseudoinverse_certificate(
            self,
            &PseudoinverseCertificate {
                inverse: inverse.clone(),
            },
        )?;
        Ok(inverse)
    }
}

/// Candidate for an exact-rational Moore-Penrose pseudoinverse.
///
/// This is a same-crate reference-checked result, not a portable proof-kernel
/// certificate or a claim that the full matrix compatibility gate has passed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PseudoinverseCertificate {
    pub inverse: Matrix,
}

// The checker deliberately does not use the candidate generator's RREF,
// solves, matrix multiplication, or expression simplifier.
#[derive(PartialEq, Eq)]
struct RationalMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<BigRational>,
}

impl RationalMatrix {
    fn from_matrix(matrix: &Matrix) -> Result<Self, MatrixError> {
        matrix.validate_certificate_domain()?;
        let entries = matrix
            .data
            .iter()
            .map(|entry| numeric_value(entry).ok_or(MatrixError::UnsupportedCertificateDomain))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rows: matrix.rows,
            cols: matrix.cols,
            entries,
        })
    }

    fn get(&self, row: usize, column: usize) -> &BigRational {
        &self.entries[row * self.cols + column]
    }

    fn dot(&self, other: &Self, row: usize, column: usize) -> BigRational {
        let mut sum = BigRational::zero();
        for inner in 0..self.cols {
            sum += self.get(row, inner) * other.get(inner, column);
        }
        sum
    }

    fn multiply(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let entries = Matrix::checked_element_count(self.rows, other.cols)?;
        let operations = 2 * self.rows as u128 * self.cols as u128 * other.cols as u128;
        check_reference_work(operations)?;
        let mut product = Vec::with_capacity(entries);
        for row in 0..self.rows {
            for column in 0..other.cols {
                product.push(self.dot(other, row, column));
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            entries: product,
        })
    }

    fn is_symmetric(&self) -> bool {
        self.rows == self.cols
            && (0..self.rows)
                .all(|row| (0..row).all(|column| self.get(row, column) == self.get(column, row)))
    }

    fn product_is_symmetric(&self, other: &Self) -> Result<bool, MatrixError> {
        if self.cols != other.rows || self.rows != other.cols {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        // Check the larger projector without allocating its square matrix.
        // This keeps tall/wide, low-rank inputs useful within the work bound.
        let pairs = self.rows as u128 * self.rows.saturating_sub(1) as u128 / 2;
        check_reference_work(4 * pairs * self.cols as u128)?;
        for row in 0..self.rows {
            for column in 0..row {
                if self.dot(other, row, column) != self.dot(other, column, row) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

fn check_reference_work(operations: u128) -> Result<(), MatrixError> {
    if operations > MAX_MATRIX_MULTIPLICATION_OPS {
        return Err(MatrixError::ResourceLimit(format!(
            "pseudoinverse verification exceeds the operation limit of {MAX_MATRIX_MULTIPLICATION_OPS}"
        )));
    }
    Ok(())
}

/// Checks `A X A = A`, `X A X = X`, `(A X)^T = A X`, and `(X A)^T = X A`.
///
/// All arithmetic is exact over the rationals; transpose therefore equals
/// conjugate transpose. Each identity is necessary: a generalized inverse
/// satisfying only the first one or two identities is not enough.
pub fn verify_pseudoinverse_certificate(
    matrix: &Matrix,
    cert: &PseudoinverseCertificate,
) -> Result<(), MatrixError> {
    matrix.validate_certificate_domain()?;
    cert.inverse.validate_certificate_domain()?;
    if matrix.rows != cert.inverse.cols || matrix.cols != cert.inverse.rows {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows,
            matrix.cols,
            cert.inverse.rows,
            cert.inverse.cols,
        ));
    }
    // For a zero input, the second identity requires X = 0. This also
    // handles zero-area matrices without quadratic dimension loops.
    if matrix.data.iter().all(|entry| entry.is_zero()) {
        return if cert.inverse.data.iter().all(|entry| entry.is_zero()) {
            Ok(())
        } else {
            Err(MatrixError::InvalidCertificate("X A X != X".to_string()))
        };
    }

    let a = RationalMatrix::from_matrix(matrix)?;
    let x = RationalMatrix::from_matrix(&cert.inverse)?;
    let (axa, xax, ax_symmetric, xa_symmetric) = if a.rows <= a.cols {
        let ax = a.multiply(&x)?;
        (
            ax.multiply(&a)?,
            x.multiply(&ax)?,
            ax.is_symmetric(),
            x.product_is_symmetric(&a)?,
        )
    } else {
        let xa = x.multiply(&a)?;
        (
            a.multiply(&xa)?,
            xa.multiply(&x)?,
            a.product_is_symmetric(&x)?,
            xa.is_symmetric(),
        )
    };
    for (holds, identity) in [
        (axa == a, "A X A != A"),
        (xax == x, "X A X != X"),
        (ax_symmetric, "A X is not symmetric"),
        (xa_symmetric, "X A is not symmetric"),
    ] {
        if !holds {
            return Err(MatrixError::InvalidCertificate(identity.to_string()));
        }
    }
    Ok(())
}
