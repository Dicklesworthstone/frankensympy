//! # fsym-matrices
//!
//! Symbolic matrix operations, determinants, eigenvalues, matrix calculus, and decompositions.

#![forbid(unsafe_code)]

use fsym_core::{BigInt, BigRational, Expr};
use fsym_polys::UnivariatePoly;
use fsym_simplify::simplify;
use fsym_solvers::{SolverError, solve_poly};
use num_traits::{One, Zero};
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
    #[error(
        "Characteristic polynomial has symbolic coefficients; exact eigenvalues require numeric entries"
    )]
    SymbolicCharacteristicPolynomial,
    #[error("Exact eigenvalues unsupported for characteristic polynomial degree {0}")]
    EigenvaluesUnsupportedDegree(usize),
    #[error("Underlying solver failure: {0}")]
    Solver(String),
}

/// Numeric value of an expression when it is fully constant.
fn numeric_value(entry: &Expr) -> Option<BigRational> {
    match entry {
        Expr::Integer(i) => Some(BigRational::from_integer(i.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Canonical expression for an exact rational: integers stay integers.
fn from_rational(r: BigRational) -> Expr {
    if r.is_integer() {
        Expr::Integer(r.to_integer())
    } else {
        Expr::Rational(r)
    }
}

fn is_perfect_square(n: &BigInt) -> Option<BigInt> {
    if n < &BigInt::zero() {
        return None;
    }
    let root = n.sqrt();
    if &root * &root == *n {
        Some(root)
    } else {
        None
    }
}

fn perfect_square_root(r: &BigRational) -> Option<BigRational> {
    Some(BigRational::new(
        is_perfect_square(r.numer())?,
        is_perfect_square(r.denom())?,
    ))
}

/// Folds provably numeric subexpressions into canonical form: sums,
/// products, and square roots of perfect-square rationals collapse;
/// everything else survives unchanged as an exact expression.
fn exact_fold(expr: &Expr) -> Expr {
    match expr {
        Expr::Add(terms) => {
            let mut sum = BigRational::zero();
            let mut rest: Vec<Expr> = Vec::new();
            for t in terms {
                let folded = exact_fold(t);
                match numeric_value(&folded) {
                    Some(v) => sum += v,
                    None => rest.push(folded),
                }
            }
            if rest.is_empty() {
                from_rational(sum)
            } else {
                if sum != BigRational::zero() {
                    rest.push(from_rational(sum));
                }
                if rest.len() == 1 {
                    rest.pop().unwrap()
                } else {
                    Expr::Add(rest)
                }
            }
        }
        Expr::Mul(factors) => {
            let mut product = BigRational::one();
            let mut rest: Vec<Expr> = Vec::new();
            for f in factors {
                let folded = exact_fold(f);
                match numeric_value(&folded) {
                    Some(v) => product *= v,
                    None => rest.push(folded),
                }
            }
            if rest.is_empty() || product.is_zero() {
                from_rational(product)
            } else {
                if product != BigRational::one() {
                    rest.insert(0, from_rational(product));
                }
                if rest.len() == 1 {
                    rest.pop().unwrap()
                } else {
                    Expr::Mul(rest)
                }
            }
        }
        Expr::Pow(base, exponent) => {
            let b = exact_fold(base);
            let e = exact_fold(exponent);
            match (numeric_value(&b), &e) {
                (Some(v), Expr::Rational(half))
                    if half.numer() == &BigInt::from(1) && half.denom() == &BigInt::from(2) =>
                {
                    match perfect_square_root(&v) {
                        Some(root) => from_rational(root),
                        None => Expr::Pow(
                            std::sync::Arc::new(from_rational(v)),
                            std::sync::Arc::new(Expr::Rational(half.clone())),
                        ),
                    }
                }
                (Some(v), Expr::Integer(n)) => {
                    let exp = i64::try_from(n.clone()).ok();
                    match exp {
                        Some(k) if (0..=64).contains(&k) => from_rational(v.pow(k as i32)),
                        Some(-1) => {
                            from_rational(BigRational::new(v.denom().clone(), v.numer().clone()))
                        }
                        _ => Expr::Pow(
                            std::sync::Arc::new(from_rational(v)),
                            std::sync::Arc::new(Expr::Integer(n.clone())),
                        ),
                    }
                }
                _ => Expr::Pow(std::sync::Arc::new(b), std::sync::Arc::new(e)),
            }
        }
        other => other.clone(),
    }
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
                let mut acc = Expr::from_i64(0);
                for k in 0..self.cols {
                    let a = &self.data[r * self.cols + k];
                    let b = &other.data[k * other.cols + c];
                    acc = Self::exact_add(&acc, &Self::exact_mul(a, b));
                }
                result_data.push(acc);
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

    /// Numeric value of an entry when it is fully constant.
    fn numeric(entry: &Expr) -> Option<BigRational> {
        match entry {
            Expr::Integer(i) => Some(BigRational::from_integer(i.clone())),
            Expr::Rational(r) => Some(r.clone()),
            _ => None,
        }
    }

    /// Exact `a * b`, folding to a number when both sides are numeric.
    fn exact_mul(a: &Expr, b: &Expr) -> Expr {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) => from_rational(x * y),
            _ => simplify(&Expr::Mul(vec![a.clone(), b.clone()])),
        }
    }

    /// Exact `a - b`, folding to a number when both sides are numeric.
    fn exact_sub(a: &Expr, b: &Expr) -> Expr {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) => from_rational(x - y),
            _ => simplify(&Expr::Add(vec![a.clone(), Expr::from_i64(-1), b.clone()])),
        }
    }

    /// Exact `a / b` as an expression; symbolic divisors stay as
    /// multiplication by an inverse power.
    fn exact_div(a: &Expr, b: &Expr) -> Expr {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) if !y.is_zero() => from_rational(x / y),
            (_, Some(y)) if y.is_zero() => panic!("exact_div by zero"),
            _ => simplify(&Expr::Mul(vec![
                a.clone(),
                Expr::Pow(
                    std::sync::Arc::new(b.clone()),
                    std::sync::Arc::new(Expr::from_i64(-1)),
                ),
            ])),
        }
    }

    /// Exact `a + b`, folding to a number when both sides are numeric.
    fn exact_add(a: &Expr, b: &Expr) -> Expr {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) => from_rational(x + y),
            _ => simplify(&Expr::Add(vec![a.clone(), b.clone()])),
        }
    }

    /// Returns the entry scaled by `factor`, simplified.
    fn scaled_entry(factor: Expr, entry: &Expr) -> Expr {
        Self::exact_mul(&factor, entry)
    }

    /// Cofactor C(i, j) = (-1)^(i+j) * minor(i, j).
    fn cofactor(&self, i: usize, j: usize) -> Result<Expr, MatrixError> {
        let sign = if (i + j).is_multiple_of(2) { 1 } else { -1 };
        let minor = self.minor_matrix(i, j)?.det()?;
        Ok(Self::scaled_entry(Expr::from_i64(sign), &minor))
    }

    /// Multiplicative inverse via the adjugate: `A^-1 = adj(A) / det(A)`.
    ///
    /// Singularity is decided exactly when `det` folds to a number; a
    /// symbolic determinant is treated as nonzero and carried through the
    /// entries as an exact expression identity.
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let det = self.det()?;
        if det.is_zero() {
            return Err(MatrixError::SingularMatrix);
        }
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                // Adjugate is the transpose of the cofactor matrix.
                data.push(Self::exact_div(&self.cofactor(c, r)?, &det));
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Rank via Gaussian elimination.
    ///
    /// Exact for matrices whose entries decide `is_zero()` numerically.
    /// Symbolic entries are conservatively counted as nonzero pivots, so
    /// the result is an upper bound on rank deficiency in that case —
    /// never an invented collapse of symbolic rows to zero.
    pub fn rank(&self) -> usize {
        let mut work = self.data.clone();
        let (rows, cols) = (self.rows, self.cols);
        let at = |r: usize, c: usize| r * cols + c;
        let mut pivot_row = 0;
        for col in 0..cols {
            if pivot_row >= rows {
                break;
            }
            // Find a row with a provably nonzero entry in this column.
            let mut pivot = None;
            for r in pivot_row..rows {
                if !work[at(r, col)].is_zero() {
                    pivot = Some(r);
                    break;
                }
            }
            let Some(pivot) = pivot else {
                continue;
            };
            if pivot != pivot_row {
                for c in 0..cols {
                    work.swap(at(pivot_row, c), at(pivot, c));
                }
            }
            let pivot_val = work[at(pivot_row, col)].clone();
            for r in (pivot_row + 1)..rows {
                if work[at(r, col)].is_zero() {
                    continue;
                }
                let factor = Self::exact_div(&work[at(r, col)], &pivot_val);
                for c in 0..cols {
                    let product = Self::exact_mul(&factor, &work[at(pivot_row, c)]);
                    work[at(r, c)] = Self::exact_sub(&work[at(r, c)], &product);
                }
            }
            pivot_row += 1;
        }
        pivot_row
    }

    /// Coefficients of `det(λI − A)` in ascending powers of `λ`.
    ///
    /// Uses Faddeev–LeVerrier: exact for numeric entries and valid as
    /// symbolic expressions otherwise; division by the loop index stays
    /// inside exact rationals.
    pub fn char_poly(&self) -> Result<Vec<Expr>, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let n = self.rows;
        // M starts as the identity; coefficients collected descending
        // (c_n first), reversed by callers that need ascending order.
        let mut m = Matrix::eye(n);
        let mut coeffs: Vec<Expr> = vec![Expr::from_i64(1)]; // c_n = 1
        for k in 1..=n {
            m = self.matmul(&m)?;
            let trace = m.trace()?;
            let k_expr = Expr::from_i64(k as i64);
            let c_k = Self::exact_mul(&Expr::from_i64(-1), &Self::exact_div(&trace, &k_expr));
            coeffs.push(c_k.clone());
            // M += c_k * I
            for i in 0..n {
                let idx = i * n + i;
                m.data[idx] =
                    Self::exact_sub(&m.data[idx], &Self::exact_mul(&Expr::from_i64(-1), &c_k));
            }
        }
        Ok(coeffs)
    }

    /// Eigenvalues as exact expressions when decidable.
    ///
    /// Computes the characteristic polynomial and solves it exactly for
    /// degrees up to 2. Larger degrees and symbolic characteristic
    /// coefficients are refused rather than answered incompletely.
    pub fn eigenvalues(&self) -> Result<Vec<Expr>, MatrixError> {
        let coeffs = self.char_poly()?;
        // char_poly yields descending powers (c_n first); UnivariatePoly
        // wants ascending, including the leading unit coefficient.
        let mut rationals = Vec::with_capacity(coeffs.len());
        for c in coeffs.iter().rev() {
            match c {
                Expr::Integer(i) => rationals.push(BigRational::from_integer(i.clone())),
                Expr::Rational(r) => rationals.push(r.clone()),
                _ => return Err(MatrixError::SymbolicCharacteristicPolynomial),
            }
        }
        let gen_sym = fsym_core::Symbol::new("lambda");
        let poly = UnivariatePoly::new(gen_sym, rationals);
        match poly.degree() {
            None | Some(0) => Err(MatrixError::SingularMatrix),
            Some(d) if d > 2 => Err(MatrixError::EigenvaluesUnsupportedDegree(d)),
            Some(_) => {
                let roots = solve_poly(&poly).map_err(|err| match err {
                    SolverError::UnsupportedDegree(d) => {
                        MatrixError::EigenvaluesUnsupportedDegree(d)
                    }
                    other => MatrixError::Solver(other.to_string()),
                })?;
                Ok(roots.iter().map(exact_fold).collect())
            }
        }
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

    fn num(n: i64) -> Expr {
        Expr::from_i64(n)
    }

    #[test]
    fn numeric_inverse_times_original_is_identity() {
        // [[2, 0], [0, 4]]^-1 = [[1/2, 0], [0, 1/4]]
        let m = Matrix::new(2, 2, vec![num(2), num(0), num(0), num(4)]).unwrap();
        let inv = m.inverse().unwrap();
        assert_eq!(
            inv.get(0, 0).unwrap(),
            &Expr::Rational(BigRational::new(1.into(), 2.into()))
        );
        assert_eq!(
            inv.get(1, 1).unwrap(),
            &Expr::Rational(BigRational::new(1.into(), 4.into()))
        );
        assert_eq!(inv.get(0, 1).unwrap(), &num(0));

        let product = inv.matmul(&m).unwrap();
        assert_eq!(*product.get(0, 0).unwrap(), num(1));
        assert_eq!(*product.get(1, 1).unwrap(), num(1));
        assert_eq!(*product.get(0, 1).unwrap(), num(0));
        assert_eq!(*product.get(1, 0).unwrap(), num(0));
    }

    #[test]
    fn singular_numeric_matrix_is_refused() {
        let m = Matrix::new(2, 2, vec![num(1), num(2), num(2), num(4)]).unwrap();
        assert_eq!(m.inverse().unwrap_err(), MatrixError::SingularMatrix);
    }

    #[test]
    fn non_square_inverse_and_charpoly_refused() {
        let m = Matrix::new(2, 3, vec![num(1), num(2), num(3), num(4), num(5), num(6)]).unwrap();
        assert_eq!(m.inverse().unwrap_err(), MatrixError::NotSquare(2, 3));
        assert_eq!(m.char_poly().unwrap_err(), MatrixError::NotSquare(2, 3));
    }

    #[test]
    fn rank_matches_independent_row_count() {
        assert_eq!(Matrix::eye(4).rank(), 4);
        assert_eq!(Matrix::zeros(2, 3).rank(), 0);

        // Third row = row1 + row2 => rank 2.
        let m = Matrix::new(
            3,
            3,
            vec![
                num(1),
                num(2),
                num(3),
                num(4),
                num(5),
                num(6),
                num(5),
                num(7),
                num(9),
            ],
        )
        .unwrap();
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn char_poly_of_diagonal_factors_exactly() {
        // det(lambda*I - diag(2, 3)) = (lambda - 2)(lambda - 3)
        //   = lambda^2 - 5*lambda + 6
        let m = Matrix::new(2, 2, vec![num(2), num(0), num(0), num(3)]).unwrap();
        let coeffs = m.char_poly().unwrap();
        let as_i64 = |e: &Expr| match e {
            Expr::Integer(i) => i64::try_from(i.clone()).ok(),
            _ => None,
        };
        let values: Vec<i64> = coeffs
            .iter()
            .map(|c| as_i64(c).expect("integer coeff"))
            .collect();
        assert_eq!(values, vec![1, -5, 6]);
    }

    #[test]
    fn eigenvalues_of_diagonal_matrix_are_exact() {
        let m = Matrix::new(2, 2, vec![num(2), num(0), num(0), num(3)]).unwrap();
        let mut eig = m.eigenvalues().unwrap();
        eig.sort_by_key(|e| match e {
            Expr::Integer(i) => i64::try_from(i.clone()).unwrap_or(i128::MAX as i64),
            _ => i64::MIN,
        });
        assert_eq!(eig, vec![num(2), num(3)]);
    }

    #[test]
    fn eigenvalues_of_rotation_are_symbolic_roots() {
        // [[0, 1], [-1, 0]] has char poly lambda^2 + 1 => roots ±sqrt(-1).
        let m = Matrix::new(2, 2, vec![num(0), num(1), num(-1), num(0)]).unwrap();
        let eig = m.eigenvalues().unwrap();
        assert_eq!(eig.len(), 2);
        for root in &eig {
            match root {
                Expr::Mul(parts) => assert_eq!(parts.len(), 2, "(-b +- sqrt(d)) / (2a) shape"),
                other => panic!("expected quadratic-formula expression, got {other:?}"),
            }
        }
    }

    #[test]
    fn eigenvalues_beyond_quadratic_degree_refused() {
        let m = Matrix::eye(3);
        assert_eq!(
            m.eigenvalues().unwrap_err(),
            MatrixError::EigenvaluesUnsupportedDegree(3)
        );
    }

    #[test]
    fn symbolic_characteristic_polynomial_refuses_eigenvalues() {
        let m = Matrix::new(
            2,
            2,
            vec![Expr::symbol("a"), num(0), num(0), Expr::symbol("d")],
        )
        .unwrap();
        assert_eq!(
            m.eigenvalues().unwrap_err(),
            MatrixError::SymbolicCharacteristicPolynomial
        );
        // The polynomial itself remains available symbolically.
        let coeffs = m.char_poly().unwrap();
        assert_eq!(coeffs.len(), 3);
    }
}
