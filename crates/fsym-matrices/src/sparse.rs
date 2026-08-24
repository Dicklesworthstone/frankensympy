//! Sparse matrix representation over $\mathbb{Q}$ and symbolic expressions (WS10).

#![forbid(unsafe_code)]

use crate::{Matrix, MatrixError};
use fsym_core::Expr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Exact sparse matrix stored as a sorted map of non-zero entries: `(row, col) -> Expr`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: BTreeMap<(usize, usize), Expr>,
}

impl SparseMatrix {
    /// Create a new sparse matrix from non-zero entries.
    pub fn new(rows: usize, cols: usize, raw_entries: BTreeMap<(usize, usize), Expr>) -> Self {
        let mut entries = BTreeMap::new();
        for ((r, c), val) in raw_entries {
            if r < rows && c < cols && !val.is_zero() {
                entries.insert((r, c), val);
            }
        }
        Self {
            rows,
            cols,
            entries,
        }
    }

    /// Construct a sparse zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: BTreeMap::new(),
        }
    }

    /// Construct a sparse identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut entries = BTreeMap::new();
        for i in 0..n {
            entries.insert((i, i), Expr::from_i64(1));
        }
        Self {
            rows: n,
            cols: n,
            entries,
        }
    }

    /// Get element at (r, c).
    pub fn get(&self, r: usize, c: usize) -> Result<Expr, MatrixError> {
        if r >= self.rows || c >= self.cols {
            return Err(MatrixError::OutOfBounds(r, c, self.rows, self.cols));
        }
        Ok(self
            .entries
            .get(&(r, c))
            .cloned()
            .unwrap_or_else(|| Expr::from_i64(0)))
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> Matrix {
        let mut data = vec![Expr::from_i64(0); self.rows * self.cols];
        for (&(r, c), val) in &self.entries {
            data[r * self.cols + c] = val.clone();
        }
        Matrix::new(self.rows, self.cols, data).expect("valid shape")
    }

    /// Matrix addition for sparse matrices.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let mut new_entries = self.entries.clone();
        for (&(r, c), val) in &other.entries {
            let entry = new_entries
                .entry((r, c))
                .or_insert_with(|| Expr::from_i64(0));
            *entry = Matrix::exact_add(entry, val);
            if entry.is_zero() {
                new_entries.remove(&(r, c));
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            entries: new_entries,
        })
    }

    /// Matrix multiplication for sparse matrices.
    pub fn matmul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let mut result_entries = BTreeMap::new();
        for (&(r, k1), v1) in &self.entries {
            for c in 0..other.cols {
                if let Some(v2) = other.entries.get(&(k1, c)) {
                    let prod = Matrix::exact_mul(v1, v2);
                    let acc = result_entries
                        .entry((r, c))
                        .or_insert_with(|| Expr::from_i64(0));
                    *acc = Matrix::exact_add(acc, &prod);
                    if acc.is_zero() {
                        result_entries.remove(&(r, c));
                    }
                }
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            entries: result_entries,
        })
    }
}
