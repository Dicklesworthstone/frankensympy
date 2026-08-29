//! PyO3 bindings exposing `fsym_matrices::Matrix` to Python (WS05, WS10).

#![forbid(unsafe_code)]

use fsym_core::Expr;
use fsym_matrices::{Matrix, MatrixError};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::expr::PyExpr;

fn matrix_err(err: MatrixError) -> PyErr {
    match err {
        MatrixError::OutOfBounds(r, c, rows, cols) => PyIndexError::new_err(format!(
            "Index ({r}, {c}) out of bounds for shape {rows}x{cols}"
        )),
        MatrixError::ShapeMismatch(r1, c1, r2, c2) => {
            PyValueError::new_err(format!("Shape mismatch: {r1}x{c1} vs {r2}x{c2}"))
        }
        MatrixError::NotSquare(r, c) => {
            PyValueError::new_err(format!("Matrix must be square: shape is {r}x{c}"))
        }
        MatrixError::SingularMatrix => PyValueError::new_err("Matrix is singular"),
        MatrixError::DivisionByZero => PyValueError::new_err("Exact matrix division by zero"),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Native matrix type exposing exact algebraic matrix operations.
#[pyclass(name = "Matrix", module = "fsym_python")]
#[derive(Clone)]
pub struct PyMatrix {
    pub(crate) inner: Matrix,
}

#[pymethods]
impl PyMatrix {
    /// Construct a new Matrix with the given dimensions and flat element sequence.
    #[new]
    #[pyo3(signature = (rows, cols, data))]
    pub fn new(rows: usize, cols: usize, data: Vec<PyExpr>) -> PyResult<Self> {
        let raw: Vec<Expr> = data.into_iter().map(|e| e.inner).collect();
        let m = Matrix::new(rows, cols, raw).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Construct an identity matrix of size N x N.
    #[staticmethod]
    pub fn eye(n: usize) -> PyResult<Self> {
        let m = Matrix::eye(n).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Construct a zero matrix of size `rows x cols`.
    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        let m = Matrix::zeros(rows, cols).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Construct a diagonal matrix from the given diagonal elements.
    #[staticmethod]
    pub fn diag(entries: Vec<PyExpr>) -> PyResult<Self> {
        let raw: Vec<Expr> = entries.into_iter().map(|e| e.inner).collect();
        let m = Matrix::diag(raw).map_err(matrix_err)?;
        Ok(Self { inner: m })
    }

    /// Matrix shape as a `(rows, cols)` tuple.
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.rows(), self.inner.cols())
    }

    /// Number of rows.
    #[getter]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns.
    #[getter]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Number of elements.
    pub fn __len__(&self) -> usize {
        self.inner.rows() * self.inner.cols()
    }

    /// Whether the matrix is square.
    #[getter]
    pub fn is_square(&self) -> bool {
        self.inner.rows() == self.inner.cols()
    }

    /// Whether the matrix is symmetric.
    #[getter]
    pub fn is_symmetric(&self) -> bool {
        self.inner.is_symmetric()
    }

    /// Whether the matrix is diagonal.
    #[getter]
    pub fn is_diagonal(&self) -> bool {
        self.inner.is_diagonal()
    }

    /// Whether the matrix is upper triangular.
    #[getter]
    pub fn is_upper_triangular(&self) -> bool {
        self.inner.is_upper_triangular()
    }

    /// Whether the matrix is lower triangular.
    #[getter]
    pub fn is_lower_triangular(&self) -> bool {
        self.inner.is_lower_triangular()
    }

    /// Element indexing: `m[r, c]`.
    pub fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        if let Ok(tuple) = key.cast::<PyTuple>()
            && tuple.len() == 2
        {
            let r: usize = tuple.get_item(0)?.extract()?;
            let c: usize = tuple.get_item(1)?.extract()?;
            let entry = self.inner.get(r, c).map_err(matrix_err)?;
            return Ok(PyExpr::from_expr(entry.clone()));
        }
        if let Ok(flat_idx) = key.extract::<usize>() {
            let cols = self.inner.cols();
            if cols == 0 {
                return Err(PyIndexError::new_err("matrix has 0 columns"));
            }
            let r = flat_idx / cols;
            let c = flat_idx % cols;
            let entry = self.inner.get(r, c).map_err(matrix_err)?;
            return Ok(PyExpr::from_expr(entry.clone()));
        }
        Err(PyTypeError::new_err(
            "Matrix indices must be integers or (row, col) integer pairs",
        ))
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    /// Transpose property shorthand `T`.
    #[getter]
    pub fn t(&self) -> Self {
        self.transpose()
    }

    /// Matrix trace: sum of diagonal elements.
    pub fn trace(&self) -> PyResult<PyExpr> {
        let tr = self.inner.trace().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(tr))
    }

    /// Matrix determinant.
    pub fn det(&self) -> PyResult<PyExpr> {
        let d = self.inner.det().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(d))
    }

    /// Matrix inverse: A^-1.
    pub fn inv(&self) -> PyResult<Self> {
        let inv_m = self.inner.inverse().map_err(matrix_err)?;
        Ok(Self { inner: inv_m })
    }

    /// Matrix inverse alias.
    pub fn inverse(&self) -> PyResult<Self> {
        self.inv()
    }

    /// Adjugate (classical adjoint) matrix.
    pub fn adjugate(&self) -> PyResult<Self> {
        let adj = self.inner.adjugate().map_err(matrix_err)?;
        Ok(Self { inner: adj })
    }

    /// Cofactor C(i, j).
    pub fn cofactor(&self, i: usize, j: usize) -> PyResult<PyExpr> {
        let c = self.inner.cofactor(i, j).map_err(matrix_err)?;
        Ok(PyExpr::from_expr(c))
    }

    /// Squared Frobenius norm.
    pub fn frobenius_norm_squared(&self) -> PyResult<PyExpr> {
        let norm_sq = self.inner.frobenius_norm_squared().map_err(matrix_err)?;
        Ok(PyExpr::from_expr(norm_sq))
    }

    /// Matrix rank.
    pub fn rank(&self) -> PyResult<usize> {
        self.inner.rank().map_err(matrix_err)
    }

    /// Reduced row echelon form (RREF) returning `(rref_matrix, pivot_columns)`.
    pub fn rref(&self) -> PyResult<(Self, Vec<usize>)> {
        let (rref_m, pivots) = self.inner.rref().map_err(matrix_err)?;
        Ok((Self { inner: rref_m }, pivots))
    }

    /// Basis vectors for the nullspace (kernel).
    pub fn nullspace(&self) -> PyResult<Vec<Self>> {
        let bases = self.inner.nullspace().map_err(matrix_err)?;
        Ok(bases.into_iter().map(|m| Self { inner: m }).collect())
    }

    /// Exact eigenvalues of the matrix.
    pub fn eigenvalues(&self) -> PyResult<Vec<PyExpr>> {
        let evals = self.inner.eigenvalues().map_err(matrix_err)?;
        Ok(evals.into_iter().map(PyExpr::from_expr).collect())
    }

    /// Matrix addition.
    pub fn __add__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.add(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Matrix subtraction.
    pub fn __sub__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.sub(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Matrix multiplication: self @ other.
    pub fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.matmul(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Scalar or matrix multiplication: self * other.
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_mat) = other.extract::<PyRef<Self>>() {
            let res = self.inner.matmul(&other_mat.inner).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(scalar_expr) = other.extract::<PyRef<PyExpr>>() {
            let res = self
                .inner
                .scalar_mul(&scalar_expr.inner)
                .map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        if let Ok(int_val) = other.extract::<i64>() {
            let scalar = Expr::from_i64(int_val);
            let res = self.inner.scalar_mul(&scalar).map_err(matrix_err)?;
            return Ok(Self { inner: res });
        }
        Err(PyTypeError::new_err(
            "Multiplication unsupported between Matrix and the given operand",
        ))
    }

    /// Matrix integer power: self ** n.
    pub fn __pow__(&self, n: usize, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let res = self.inner.pow(n).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Hadamard (elementwise) product.
    pub fn hadamard(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.hadamard(&other.inner).map_err(matrix_err)?;
        Ok(Self { inner: res })
    }

    /// Returns a flat list of all entries.
    pub fn flat(&self) -> Vec<PyExpr> {
        self.inner
            .data()
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    /// Returns the matrix as nested rows: `List[List[Expr]]`.
    pub fn to_list(&self) -> Vec<Vec<PyExpr>> {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        let data = self.inner.data();
        let mut result = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(PyExpr::from_expr(data[r * cols + c].clone()));
            }
            result.push(row);
        }
        result
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Matrix({}x{}, {:?})",
            self.inner.rows(),
            self.inner.cols(),
            self.inner.data()
        )
    }

    pub fn __str__(&self) -> String {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        let data = self.inner.data();
        let mut lines = Vec::with_capacity(rows);
        for r in 0..rows {
            let row_strs: Vec<String> = (0..cols)
                .map(|c| format!("{}", data[r * cols + c]))
                .collect();
            lines.push(format!("[{}]", row_strs.join(", ")));
        }
        format!("Matrix([\n  {}\n])", lines.join(",\n  "))
    }

    pub fn _repr_latex_(&self) -> PyResult<String> {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        let data = self.inner.data();
        let mut row_strs = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut entries = Vec::with_capacity(cols);
            for c in 0..cols {
                let s = fsym_printing::latex(&data[r * cols + c])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                entries.push(s);
            }
            row_strs.push(entries.join(" & "));
        }
        Ok(format!(
            "\\left[\\begin{{matrix}}{}\\end{{matrix}}\\right]",
            row_strs.join(" \\\\ ")
        ))
    }
}
