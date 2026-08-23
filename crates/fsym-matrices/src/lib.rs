//! # fsym-matrices
//!
//! Symbolic matrix operations, determinants, eigenvalues, matrix calculus, and decompositions.

#![forbid(unsafe_code)]

use fsym_core::Expr;
use fsym_simplify::simplify;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MatrixError {
    #[error("Dimension mismatch: cannot multiply {0}x{1} by {2}x{3}")]
    ShapeMismatch(usize, usize, usize, usize),
    #[error("Matrix must be square to compute determinant or inverse: shape is {0}x{1}")]
    NotSquare(usize, usize),
    #[error("Singular matrix cannot be inverted")]
    SingularMatrix,
    #[error("Index out of bounds ({0}, {1}) for shape {2}x{3}")]
    OutOfBounds(usize, usize, usize, usize),
}

/// 2D Symbolic Matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Expr>,
}

impl Matrix {
    /// Create a new matrix with shape `(rows, cols)` and flat elements.
    pub fn new(rows: usize, cols: usize, data: Vec<Expr>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::ShapeMismatch(rows, cols, 0, data.len()));
        }
        Ok(Self { rows, cols, data })
    }

    /// Create an identity matrix of size N x N.
    pub fn eye(n: usize) -> Self {
        let mut data = Vec::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                if r == c {
                    data.push(Expr::from_i64(1));
                } else {
                    data.push(Expr::from_i64(0));
                }
            }
        }
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    /// Create a zero matrix of size `rows x cols`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Expr::from_i64(0); rows * cols],
        }
    }

    /// Get element at (row, col).
    pub fn get(&self, r: usize, c: usize) -> Result<&Expr, MatrixError> {
        if r >= self.rows || c >= self.cols {
            return Err(MatrixError::OutOfBounds(r, c, self.rows, self.cols));
        }
        Ok(&self.data[r * self.cols + c])
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                data.push(self.data[r * self.cols + c].clone());
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Matrix trace: sum of diagonal elements.
    pub fn trace(&self) -> Result<Expr, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let mut diag = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            diag.push(self.data[i * self.cols + i].clone());
        }
        Ok(simplify(&Expr::Add(diag)))
    }

    /// Matrix multiplication: self * other.
    pub fn matmul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let mut result_data = Vec::with_capacity(self.rows * other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum_terms = Vec::with_capacity(self.cols);
                for k in 0..self.cols {
                    let a = &self.data[r * self.cols + k];
                    let b = &other.data[k * other.cols + c];
                    sum_terms.push(Expr::Mul(vec![a.clone(), b.clone()]));
                }
                result_data.push(simplify(&Expr::Add(sum_terms)));
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        })
    }

    /// Determinant computation for square matrix.
    pub fn det(&self) -> Result<Expr, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        match self.rows {
            1 => Ok(self.data[0].clone()),
            2 => {
                // ad - bc
                let a = &self.data[0];
                let b = &self.data[1];
                let c = &self.data[2];
                let d = &self.data[3];
                let ad = Expr::Mul(vec![a.clone(), d.clone()]);
                let bc = Expr::Mul(vec![Expr::from_i64(-1), b.clone(), c.clone()]);
                Ok(simplify(&Expr::Add(vec![ad, bc])))
            }
            n => {
                // Laplace expansion along row 0
                let mut terms = Vec::new();
                for c in 0..n {
                    let elem = &self.data[c];
                    if elem.is_zero() {
                        continue;
                    }
                    let sub = self.minor_matrix(0, c)?;
                    let sub_det = sub.det()?;
                    let sign = if c % 2 == 0 { 1 } else { -1 };
                    terms.push(Expr::Mul(vec![Expr::from_i64(sign), elem.clone(), sub_det]));
                }
                Ok(simplify(&Expr::Add(terms)))
            }
        }
    }

    /// Extract submatrix by removing given row and col.
    pub fn minor_matrix(&self, rem_r: usize, rem_c: usize) -> Result<Self, MatrixError> {
        if rem_r >= self.rows || rem_c >= self.cols {
            return Err(MatrixError::OutOfBounds(rem_r, rem_c, self.rows, self.cols));
        }
        let mut new_data = Vec::with_capacity((self.rows - 1) * (self.cols - 1));
        for r in 0..self.rows {
            if r == rem_r {
                continue;
            }
            for c in 0..self.cols {
                if c == rem_c {
                    continue;
                }
                new_data.push(self.data[r * self.cols + c].clone());
            }
        }
        Ok(Self {
            rows: self.rows - 1,
            cols: self.cols - 1,
            data: new_data,
        })
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix({}x{}):", self.rows, self.cols)?;
        for r in 0..self.rows {
            write!(f, "  [")?;
            for c in 0..self.cols {
                write!(f, " {}", self.data[r * self.cols + c])?;
                if c + 1 < self.cols {
                    write!(f, ",")?;
                }
            }
            writeln!(f, " ]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_det_2x2() {
        // [ [a, b], [c, d] ] => ad - bc
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let c = Expr::symbol("c");
        let d = Expr::symbol("d");
        let m = Matrix::new(2, 2, vec![a.clone(), b.clone(), c.clone(), d.clone()]).unwrap();
        let det = m.det().unwrap();
        assert_eq!(
            det,
            Expr::Add(vec![
                Expr::Mul(vec![a, d]),
                Expr::Mul(vec![Expr::from_i64(-1), b, c]),
            ])
        );
    }

    #[test]
    fn test_matrix_eye_matmul() {
        let eye = Matrix::eye(3);
        let m = Matrix::new(
            3,
            3,
            vec![
                Expr::from_i64(1),
                Expr::from_i64(2),
                Expr::from_i64(3),
                Expr::from_i64(4),
                Expr::from_i64(5),
                Expr::from_i64(6),
                Expr::from_i64(7),
                Expr::from_i64(8),
                Expr::from_i64(9),
            ],
        )
        .unwrap();
        let res = m.matmul(&eye).unwrap();
        assert_eq!(res.data, m.data);
    }
}
