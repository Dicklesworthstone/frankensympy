//! # fsym-matrices
//!
//! Symbolic matrix operations, determinants, eigenvalues, matrix calculus, and decompositions.

#![forbid(unsafe_code)]

pub mod sparse;
pub use sparse::*;

use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use fsym_simplify::simplify;
use fsym_solvers::{SolverError, solve_poly};
use num_traits::{One, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use thiserror::Error;

const MATRIX_SCHEMA_VERSION: u32 = 1;
const MAX_MATRIX_ENTRIES: usize = 262_144;
const MAX_MATRIX_DIMENSION: usize = MAX_MATRIX_ENTRIES;
const MAX_DETERMINANT_DIMENSION: usize = 8;
const MAX_CHARACTERISTIC_POLYNOMIAL_DIMENSION: usize = 32;
const MAX_MATRIX_MULTIPLICATION_OPS: u128 = 10_000_000;
const MAX_MATRIX_POLYNOMIAL_EVALUATION_OPS: u128 = 10_000_000;
const MAX_RREF_OPS: u128 = 10_000_000;
const MAX_NULLSPACE_BASIS_ENTRIES: usize = MAX_MATRIX_ENTRIES;
// floor(sqrt(MAX_NULLSPACE_BASIS_ENTRIES)): a full nullspace basis has at
// least `vectors * vectors` scalar entries because nullity <= column count.
const MAX_NULLSPACE_BASIS_VECTORS: usize = 512;

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
    #[error("Matrix shape {0}x{1} overflows the addressable element count")]
    ShapeOverflow(usize, usize),
    #[error("Matrix shape {0}x{1} requires {2} entries, but storage contains {3}")]
    InvalidStorageLength(usize, usize, usize, usize),
    #[error("Matrix has {0} entries, exceeding the limit of {MAX_MATRIX_ENTRIES}")]
    EntryLimitExceeded(usize),
    #[error("Matrix shape {0}x{1} has a dimension exceeding the limit of {MAX_MATRIX_DIMENSION}")]
    DimensionLimitExceeded(usize, usize),
    #[error("Matrix operation exceeds a supported resource bound: {0}")]
    ResourceLimit(String),
    #[error("A symbolic pivot or determinant may be zero; an unconditional result is unsafe")]
    SymbolicZeroUndetermined,
    #[error("Exact matrix division by zero")]
    DivisionByZero,
    #[error("Matrix certificate verification currently supports exact rational entries only")]
    UnsupportedCertificateDomain,
    #[error("Matrix certificate rejected: {0}")]
    InvalidCertificate(String),
    #[error("Invalid polynomial for matrix evaluation: {0}")]
    InvalidPolynomial(String),
    #[error("LU certificate matrix P is not a 0/1 permutation matrix")]
    InvalidPermutationMatrix,
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
    let root = n.sqrt()?;
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

fn check_nullspace_basis_size(columns: usize, vectors: usize) -> Result<usize, MatrixError> {
    let entries = columns.checked_mul(vectors).ok_or_else(|| {
        MatrixError::ResourceLimit("nullspace basis entry count overflowed".to_string())
    })?;
    if entries > MAX_NULLSPACE_BASIS_ENTRIES || vectors > MAX_NULLSPACE_BASIS_VECTORS {
        return Err(MatrixError::ResourceLimit(format!(
            "nullspace basis exceeds the limits of {MAX_NULLSPACE_BASIS_ENTRIES} aggregate entries and {MAX_NULLSPACE_BASIS_VECTORS} vectors"
        )));
    }
    Ok(entries)
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
                        Some(k) if (0..=64).contains(&k) => match v.pow(k as i32) {
                            Ok(power) => from_rational(power),
                            Err(_) => Expr::Pow(
                                std::sync::Arc::new(from_rational(v)),
                                std::sync::Arc::new(Expr::Integer(n.clone())),
                            ),
                        },
                        Some(-1) => match v.pow(-1) {
                            Ok(power) => from_rational(power),
                            Err(_) => Expr::Pow(
                                std::sync::Arc::new(from_rational(v)),
                                std::sync::Arc::new(Expr::Integer(n.clone())),
                            ),
                        },
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Expr>,
}

#[derive(Serialize)]
struct MatrixWireRef<'a> {
    schema_version: u32,
    rows: usize,
    cols: usize,
    data: &'a [Expr],
}

struct BoundedExprVec(Vec<Expr>);

impl<'de> Deserialize<'de> for BoundedExprVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BoundedExprVecVisitor;

        impl<'de> Visitor<'de> for BoundedExprVecVisitor {
            type Value = BoundedExprVec;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "an expression sequence with at most {MAX_MATRIX_ENTRIES} entries"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let hinted = sequence.size_hint().unwrap_or(0);
                if hinted > MAX_MATRIX_ENTRIES {
                    return Err(serde::de::Error::custom(format!(
                        "matrix data exceeds the entry limit of {MAX_MATRIX_ENTRIES}"
                    )));
                }
                let mut data = Vec::with_capacity(hinted.min(MAX_MATRIX_ENTRIES));
                loop {
                    if data.len() == MAX_MATRIX_ENTRIES {
                        if sequence.next_element::<IgnoredAny>()?.is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "matrix data exceeds the entry limit of {MAX_MATRIX_ENTRIES}"
                            )));
                        }
                        break;
                    }
                    let Some(entry) = sequence.next_element()? else {
                        break;
                    };
                    data.push(entry);
                }
                Ok(BoundedExprVec(data))
            }
        }

        deserializer.deserialize_seq(BoundedExprVecVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixWire {
    schema_version: u32,
    rows: usize,
    cols: usize,
    data: BoundedExprVec,
}

impl Serialize for Matrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate_shape().map_err(serde::ser::Error::custom)?;
        MatrixWireRef {
            schema_version: MATRIX_SCHEMA_VERSION,
            rows: self.rows,
            cols: self.cols,
            data: &self.data,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Matrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MatrixWire::deserialize(deserializer)?;
        if wire.schema_version != MATRIX_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported matrix schema version {}",
                wire.schema_version
            )));
        }
        Self::new(wire.rows, wire.cols, wire.data.0).map_err(serde::de::Error::custom)
    }
}

impl Matrix {
    fn checked_element_count(rows: usize, cols: usize) -> Result<usize, MatrixError> {
        let entries = rows
            .checked_mul(cols)
            .ok_or(MatrixError::ShapeOverflow(rows, cols))?;
        if entries > MAX_MATRIX_ENTRIES {
            return Err(MatrixError::EntryLimitExceeded(entries));
        }
        // A product bound alone does not constrain a zero-area shape: `0 x usize::MAX`
        // contains no entries but can still drive effectively unbounded dimension loops.
        if rows > MAX_MATRIX_DIMENSION || cols > MAX_MATRIX_DIMENSION {
            return Err(MatrixError::DimensionLimitExceeded(rows, cols));
        }
        Ok(entries)
    }

    fn validate_shape(&self) -> Result<(), MatrixError> {
        let expected = Self::checked_element_count(self.rows, self.cols)?;
        if self.data.len() != expected {
            return Err(MatrixError::InvalidStorageLength(
                self.rows,
                self.cols,
                expected,
                self.data.len(),
            ));
        }
        Ok(())
    }

    fn validate_certificate_domain(&self) -> Result<(), MatrixError> {
        self.validate_shape()?;
        if self
            .data
            .iter()
            .any(|entry| !matches!(entry, Expr::Integer(_) | Expr::Rational(_)))
        {
            return Err(MatrixError::UnsupportedCertificateDomain);
        }
        Ok(())
    }

    fn charge_matrix_storage<M: fsym_budget::BudgetMeter>(
        meter: &mut M,
        entries: usize,
    ) -> Result<(), MatrixError> {
        if entries == 0 {
            return Ok(());
        }
        let bytes = entries
            .checked_mul(std::mem::size_of::<Expr>())
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| {
                MatrixError::ResourceLimit(
                    "matrix storage charge exceeds the supported u64 range".to_string(),
                )
            })?;
        meter
            .charge_batch(&[
                (fsym_budget::Dimension::MemoryBytes, bytes),
                (fsym_budget::Dimension::AllocationCount, 1),
            ])
            .map_err(|error| MatrixError::ResourceLimit(error.to_string()))
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Flat row-major entries.
    pub fn data(&self) -> &[Expr] {
        &self.data
    }

    /// Create a new matrix with shape `(rows, cols)` and flat elements.
    pub fn new(rows: usize, cols: usize, data: Vec<Expr>) -> Result<Self, MatrixError> {
        let expected = Self::checked_element_count(rows, cols)?;
        if data.len() != expected {
            return Err(MatrixError::InvalidStorageLength(
                rows,
                cols,
                expected,
                data.len(),
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// Create an identity matrix of size N x N.
    pub fn eye(n: usize) -> Result<Self, MatrixError> {
        let entries = Self::checked_element_count(n, n)?;
        let mut data = Vec::with_capacity(entries);
        for r in 0..n {
            for c in 0..n {
                if r == c {
                    data.push(Expr::from_i64(1));
                } else {
                    data.push(Expr::from_i64(0));
                }
            }
        }
        Self::new(n, n, data)
    }

    /// Create a zero matrix of size `rows x cols`.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        let entries = Self::checked_element_count(rows, cols)?;
        Self::new(rows, cols, vec![Expr::from_i64(0); entries])
    }

    /// Create a diagonal matrix of size N x N from the given diagonal entries.
    pub fn diag(entries: Vec<Expr>) -> Result<Self, MatrixError> {
        let n = entries.len();
        let total = Self::checked_element_count(n, n)?;
        let mut data = Vec::with_capacity(total);
        for (r, entry) in entries.into_iter().enumerate() {
            for c in 0..n {
                if r == c {
                    data.push(entry.clone());
                } else {
                    data.push(Expr::from_i64(0));
                }
            }
        }
        Self::new(n, n, data)
    }

    /// Checks if this matrix is symmetric ($A = A^T$).
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in 0..r {
                if self.data[r * self.cols + c] != self.data[c * self.cols + r] {
                    return false;
                }
            }
        }
        true
    }

    /// Checks if this matrix is a diagonal matrix (all non-diagonal entries are zero).
    pub fn is_diagonal(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                if r != c && !self.data[r * self.cols + c].is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Checks if this matrix is upper triangular (all entries below main diagonal are zero).
    pub fn is_upper_triangular(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in 0..r {
                if !self.data[r * self.cols + c].is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Checks if this matrix is lower triangular (all entries above main diagonal are zero).
    pub fn is_lower_triangular(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for r in 0..self.rows {
            for c in (r + 1)..self.cols {
                if !self.data[r * self.cols + c].is_zero() {
                    return false;
                }
            }
        }
        true
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
        let mut meter = fsym_budget::Unbounded;
        self.matmul_with_meter(other, &mut meter)
    }

    fn matmul_with_meter<M: fsym_budget::BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Self, MatrixError> {
        self.validate_shape()?;
        other.validate_shape()?;
        if self.cols != other.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let result_entries = Self::checked_element_count(self.rows, other.cols)?;
        let operation_count = (self.rows as u128)
            .checked_mul(self.cols as u128)
            .and_then(|value| value.checked_mul(other.cols as u128))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                MatrixError::ResourceLimit(
                    "matrix multiplication operation count overflowed".to_string(),
                )
            })?;
        if operation_count > MAX_MATRIX_MULTIPLICATION_OPS {
            return Err(MatrixError::ResourceLimit(format!(
                "matrix multiplication exceeds the operation limit of {MAX_MATRIX_MULTIPLICATION_OPS}"
            )));
        }
        meter
            .checkpoint()
            .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
        Self::charge_matrix_storage(meter, result_entries)?;
        let per_entry_work = u64::try_from(self.cols)
            .ok()
            .and_then(|inner| inner.checked_mul(2))
            .ok_or_else(|| {
                MatrixError::ResourceLimit(
                    "matrix per-entry operation count exceeds u64".to_string(),
                )
            })?;
        let mut result_data = Vec::with_capacity(result_entries);
        for r in 0..self.rows {
            meter
                .checkpoint()
                .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
            for c in 0..other.cols {
                if per_entry_work != 0 {
                    meter
                        .charge(fsym_budget::Dimension::ComputeSteps, per_entry_work)
                        .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
                }
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
        self.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        if self.rows > MAX_DETERMINANT_DIMENSION {
            return Err(MatrixError::ResourceLimit(format!(
                "Laplace determinant expansion supports dimensions up to {MAX_DETERMINANT_DIMENSION}"
            )));
        }
        match self.rows {
            0 => Ok(Expr::from_i64(1)),
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
    pub(crate) fn exact_mul(a: &Expr, b: &Expr) -> Expr {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) => from_rational(x * y),
            _ => simplify(&Expr::Mul(vec![a.clone(), b.clone()])),
        }
    }

    /// Exact `a - b`, folding to a number when both sides are numeric.
    pub(crate) fn exact_sub(a: &Expr, b: &Expr) -> Expr {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) => from_rational(x - y),
            _ => simplify(&Expr::Add(vec![
                a.clone(),
                Expr::Mul(vec![Expr::from_i64(-1), b.clone()]),
            ])),
        }
    }

    /// Exact `a / b` when the divisor is provably nonzero.
    pub(crate) fn exact_div(a: &Expr, b: &Expr) -> Result<Expr, MatrixError> {
        match (Self::numeric(a), Self::numeric(b)) {
            (Some(x), Some(y)) if !y.is_zero() => Ok(from_rational(x / y)),
            (_, Some(y)) if y.is_zero() => Err(MatrixError::DivisionByZero),
            (_, Some(_)) => Ok(simplify(&Expr::Mul(vec![
                a.clone(),
                Expr::Pow(
                    std::sync::Arc::new(b.clone()),
                    std::sync::Arc::new(Expr::from_i64(-1)),
                ),
            ]))),
            (_, None) => Err(MatrixError::SymbolicZeroUndetermined),
        }
    }

    /// Exact `a + b`, folding to a number when both sides are numeric.
    pub(crate) fn exact_add(a: &Expr, b: &Expr) -> Expr {
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
    /// Singularity must be decided exactly. Symbolic determinants are
    /// refused because an unconditional inverse would be invalid when the
    /// determinant specializes to zero.
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let det = self.det()?;
        match Self::numeric(&det) {
            Some(value) if value.is_zero() => return Err(MatrixError::SingularMatrix),
            Some(_) => {}
            None => return Err(MatrixError::SymbolicZeroUndetermined),
        }
        let mut data = Vec::with_capacity(Self::checked_element_count(self.rows, self.cols)?);
        for r in 0..self.rows {
            for c in 0..self.cols {
                // Adjugate is the transpose of the cofactor matrix.
                data.push(Self::exact_div(&self.cofactor(c, r)?, &det)?);
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Rank via exact row reduction. A pivot whose zero status depends on a
    /// symbolic specialization is refused rather than guessed nonzero.
    pub fn rank(&self) -> Result<usize, MatrixError> {
        Ok(self.rref()?.1.len())
    }

    /// Coefficients of `det(λI − A)` in descending powers of `λ`.
    ///
    /// Uses Faddeev–LeVerrier: exact for numeric entries and valid as
    /// symbolic expressions otherwise; division by the loop index stays
    /// inside exact rationals.
    pub fn char_poly(&self) -> Result<Vec<Expr>, MatrixError> {
        self.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let n = self.rows;
        if n > MAX_CHARACTERISTIC_POLYNOMIAL_DIMENSION {
            return Err(MatrixError::ResourceLimit(format!(
                "characteristic polynomial supports dimensions up to {MAX_CHARACTERISTIC_POLYNOMIAL_DIMENSION}"
            )));
        }
        // M starts as the identity; coefficients collected descending
        // (c_n first), reversed by callers that need ascending order.
        let mut m = Matrix::eye(n)?;
        let mut coeffs: Vec<Expr> = vec![Expr::from_i64(1)]; // c_n = 1
        for k in 1..=n {
            m = self.matmul(&m)?;
            let trace = m.trace()?;
            let signed_k = i64::try_from(k).map_err(|_| {
                MatrixError::ResourceLimit("matrix dimension exceeds i64".to_string())
            })?;
            let k_expr = Expr::from_i64(signed_k);
            let c_k = Self::exact_mul(&Expr::from_i64(-1), &Self::exact_div(&trace, &k_expr)?);
            coeffs.push(c_k.clone());
            // M += c_k * I
            for i in 0..n {
                let idx = i * n + i;
                m.data[idx] = Self::exact_add(&m.data[idx], &c_k);
            }
        }
        Ok(coeffs)
    }

    /// Evaluates a polynomial $P(A) = \sum_{k=0}^d c_k A^k$ on this square matrix using Horner's method.
    pub fn eval_poly(&self, poly: &UnivariatePoly) -> Result<Matrix, MatrixError> {
        self.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        poly.validate_shape()
            .map_err(|error| MatrixError::InvalidPolynomial(error.to_string()))?;
        let n = self.rows;
        if n == 0 {
            return Matrix::new(0, 0, Vec::new());
        }
        if poly.is_zero() {
            return Matrix::zeros(n, n);
        }
        let d = poly.coeffs.len() - 1;
        let matrix_multiply_ops = (n as u128)
            .checked_mul(n as u128)
            .and_then(|value| value.checked_mul(n as u128))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                MatrixError::ResourceLimit(
                    "matrix polynomial operation count overflowed".to_string(),
                )
            })?;
        let total_ops = matrix_multiply_ops
            .checked_add(n as u128)
            .and_then(|per_coefficient| per_coefficient.checked_mul(d as u128))
            .and_then(|horner_ops| horner_ops.checked_add(n as u128))
            .ok_or_else(|| {
                MatrixError::ResourceLimit(
                    "matrix polynomial operation count overflowed".to_string(),
                )
            })?;
        if total_ops > MAX_MATRIX_POLYNOMIAL_EVALUATION_OPS {
            return Err(MatrixError::ResourceLimit(format!(
                "matrix polynomial evaluation requires {total_ops} scalar operations, exceeding the limit of {MAX_MATRIX_POLYNOMIAL_EVALUATION_OPS}"
            )));
        }
        // Horner's method: start with c_d * I
        let mut res = Matrix::eye(n)?;
        let c_lead = from_rational(poly.coeffs[d].clone());
        for i in 0..n {
            res.data[i * n + i] = c_lead.clone();
        }
        for k in (0..d).rev() {
            res = res.matmul(self)?;
            let c_k = from_rational(poly.coeffs[k].clone());
            for i in 0..n {
                let idx = i * n + i;
                res.data[idx] = Self::exact_add(&res.data[idx], &c_k);
            }
        }
        Ok(res)
    }

    /// Computes the characteristic polynomial as a [`UnivariatePoly`] over $\mathbb{Q}[\lambda]$.
    pub fn char_poly_as_poly(&self, lambda_sym: &str) -> Result<UnivariatePoly, MatrixError> {
        let coeffs = self.char_poly()?;
        let mut rationals = Vec::with_capacity(coeffs.len());
        for c in coeffs.iter().rev() {
            match c {
                Expr::Integer(i) => rationals.push(BigRational::from_integer(i.clone())),
                Expr::Rational(r) => rationals.push(r.clone()),
                _ => return Err(MatrixError::SymbolicCharacteristicPolynomial),
            }
        }
        let gen_sym = Symbol::new(lambda_sym);
        Ok(UnivariatePoly::new(gen_sym, rationals))
    }

    /// Computes the characteristic polynomial and validates its certificate.
    pub fn char_poly_with_certificate(
        &self,
        lambda_sym: &str,
    ) -> Result<CharpolyCertificate, MatrixError> {
        let poly = self.char_poly_as_poly(lambda_sym)?;
        let cert = CharpolyCertificate { poly };
        verify_charpoly_certificate(self, &cert)?;
        Ok(cert)
    }

    /// Eigenvalues as exact expressions when decidable.
    ///
    /// Computes the characteristic polynomial and solves it exactly for
    /// degrees up to 2. Larger degrees and symbolic characteristic
    /// coefficients are refused rather than answered incompletely.
    pub fn eigenvalues(&self) -> Result<Vec<Expr>, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        if self.rows == 0 {
            return Ok(Vec::new());
        }
        if self.rows > 2 {
            return Err(MatrixError::EigenvaluesUnsupportedDegree(self.rows));
        }
        let poly = self.char_poly_as_poly("lambda")?;
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
    fn select_numeric_pivot(
        work: &[Expr],
        rows: usize,
        cols: usize,
        first_row: usize,
        col: usize,
    ) -> Result<Option<usize>, MatrixError> {
        let mut saw_symbolic_candidate = false;
        for row in first_row..rows {
            let entry = &work[row * cols + col];
            match Self::numeric(entry) {
                Some(value) if !value.is_zero() => return Ok(Some(row)),
                Some(_) => {}
                None if entry.is_zero() => {}
                None => saw_symbolic_candidate = true,
            }
        }
        if saw_symbolic_candidate {
            Err(MatrixError::SymbolicZeroUndetermined)
        } else {
            Ok(None)
        }
    }

    /// Computes the Reduced Row Echelon Form (RREF) and returns `(rref_matrix, pivot_columns)`.
    pub fn rref(&self) -> Result<(Matrix, Vec<usize>), MatrixError> {
        let mut meter = fsym_budget::Unbounded;
        self.rref_with_meter(&mut meter)
    }

    fn rref_with_meter<M: fsym_budget::BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<(Matrix, Vec<usize>), MatrixError> {
        self.validate_shape()?;
        let operation_bound = (self.rows as u128)
            .checked_mul(self.cols as u128)
            .and_then(|value| value.checked_mul(self.cols as u128))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                MatrixError::ResourceLimit("RREF operation bound overflowed".to_string())
            })?;
        if operation_bound > MAX_RREF_OPS {
            return Err(MatrixError::ResourceLimit(format!(
                "RREF exceeds the operation limit of {MAX_RREF_OPS}"
            )));
        }
        meter
            .checkpoint()
            .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
        Self::charge_matrix_storage(meter, self.data.len())?;
        let mut work = self.data.clone();
        let (rows, cols) = (self.rows, self.cols);
        let at = |r: usize, c: usize| r * cols + c;
        let mut pivot_row = 0;
        let mut pivot_cols = Vec::new();

        for col in 0..cols {
            meter
                .checkpoint()
                .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
            if pivot_row >= rows {
                break;
            }
            let search_work = u64::try_from(rows - pivot_row).map_err(|_| {
                MatrixError::ResourceLimit("matrix row count exceeds u64".to_string())
            })?;
            if search_work != 0 {
                meter
                    .charge(fsym_budget::Dimension::ComputeSteps, search_work)
                    .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
            }
            let pivot = Self::select_numeric_pivot(&work, rows, cols, pivot_row, col)?;
            let Some(pivot) = pivot else {
                continue;
            };
            if pivot != pivot_row {
                let swap_work = u64::try_from(cols).map_err(|_| {
                    MatrixError::ResourceLimit("matrix column count exceeds u64".to_string())
                })?;
                if swap_work != 0 {
                    meter
                        .charge(fsym_budget::Dimension::ComputeSteps, swap_work)
                        .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
                }
                for c in 0..cols {
                    work.swap(at(pivot_row, c), at(pivot, c));
                }
            }
            // Scale pivot row so pivot element is 1
            let pivot_val = work[at(pivot_row, col)].clone();
            let row_work = u64::try_from(cols).map_err(|_| {
                MatrixError::ResourceLimit("matrix column count exceeds u64".to_string())
            })?;
            let elimination_work = row_work.checked_mul(2).ok_or_else(|| {
                MatrixError::ResourceLimit("RREF row operation count exceeds u64".to_string())
            })?;
            if row_work != 0 {
                meter
                    .charge(fsym_budget::Dimension::ComputeSteps, row_work)
                    .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
            }
            for c in 0..cols {
                work[at(pivot_row, c)] = Self::exact_div(&work[at(pivot_row, c)], &pivot_val)?;
            }

            // Eliminate all other rows in this column (both above and below)
            for r in 0..rows {
                if r == pivot_row || work[at(r, col)].is_zero() {
                    continue;
                }
                meter
                    .checkpoint()
                    .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
                if elimination_work != 0 {
                    meter
                        .charge(fsym_budget::Dimension::ComputeSteps, elimination_work)
                        .map_err(|error| MatrixError::ResourceLimit(error.to_string()))?;
                }
                let factor = work[at(r, col)].clone();
                for c in 0..cols {
                    let term = Self::exact_mul(&factor, &work[at(pivot_row, c)]);
                    work[at(r, c)] = Self::exact_sub(&work[at(r, c)], &term);
                }
            }

            pivot_cols.push(col);
            pivot_row += 1;
        }

        Ok((Matrix::new(rows, cols, work)?, pivot_cols))
    }

    /// Computes basis vectors spanning the nullspace (kernel) of this matrix: $A \cdot v = 0$.
    pub fn nullspace(&self) -> Result<Vec<Matrix>, MatrixError> {
        let (rref_mat, pivot_cols) = self.rref()?;
        let n_cols = self.cols;
        let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().cloned().collect();
        let free_cols: Vec<usize> = (0..n_cols).filter(|c| !pivot_set.contains(c)).collect();
        check_nullspace_basis_size(n_cols, free_cols.len())?;

        let mut basis = Vec::with_capacity(free_cols.len());
        for &free_col in &free_cols {
            let mut vec_data = vec![Expr::from_i64(0); n_cols];
            vec_data[free_col] = Expr::from_i64(1);
            for (pivot_row_idx, &pivot_col) in pivot_cols.iter().enumerate() {
                let entry = rref_mat.get(pivot_row_idx, free_col)?;
                let neg_entry = Self::exact_mul(&Expr::from_i64(-1), entry);
                vec_data[pivot_col] = neg_entry;
            }
            basis.push(Matrix::new(n_cols, 1, vec_data)?);
        }
        Ok(basis)
    }

    /// Computes the exact Jacobian matrix for a system of expressions: $J_{i, j} = \frac{\partial f_i}{\partial x_j}$.
    pub fn jacobian(exprs: &[Expr], vars: &[Symbol]) -> Result<Matrix, MatrixError> {
        let rows = exprs.len();
        let cols = vars.len();
        let mut data = Vec::with_capacity(Self::checked_element_count(rows, cols)?);
        for expr in exprs {
            for var in vars {
                let d = fsym_calculus::diff(expr, var);
                data.push(simplify(&d));
            }
        }
        Matrix::new(rows, cols, data)
    }

    /// Convert dense matrix to sparse matrix representation.
    pub fn to_sparse(&self) -> Result<SparseMatrix, MatrixError> {
        self.validate_shape()?;
        let mut entries = std::collections::BTreeMap::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let elem = &self.data[r * self.cols + c];
                if !elem.is_zero() {
                    entries.insert((r, c), elem.clone());
                }
            }
        }
        SparseMatrix::new(self.rows, self.cols, entries)
    }

    /// Metered matrix multiplication with cancellation checkpoint and step charging.
    pub fn metered_matmul<M: fsym_budget::BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Self, MatrixError> {
        self.matmul_with_meter(other, meter)
    }

    /// Metered RREF computation with cancellation checkpoint and step charging.
    pub fn metered_rref<M: fsym_budget::BudgetMeter>(
        &self,
        meter: &mut M,
    ) -> Result<(Self, Vec<usize>), MatrixError> {
        self.rref_with_meter(meter)
    }

    /// Exact-rational LU decomposition with partial row pivoting: $P \cdot A = L \cdot U$.
    pub fn lu(&self) -> Result<LuCertificate, MatrixError> {
        self.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let n = self.rows;
        let mut p_mat = Matrix::eye(n)?;
        let mut l_mat = Matrix::eye(n)?;
        let mut u_mat = self.clone();

        let at = |r: usize, c: usize| r * n + c;

        for k in 0..n {
            let pivot_row = Self::select_numeric_pivot(&u_mat.data, n, n, k, k)?;
            let Some(p_idx) = pivot_row else {
                continue;
            };
            if p_idx != k {
                for c in 0..n {
                    u_mat.data.swap(at(k, c), at(p_idx, c));
                    p_mat.data.swap(at(k, c), at(p_idx, c));
                }
                for c in 0..k {
                    l_mat.data.swap(at(k, c), at(p_idx, c));
                }
            }
            let pivot_val = u_mat.data[at(k, k)].clone();
            if pivot_val.is_zero() {
                continue;
            }
            for i in (k + 1)..n {
                let entry_i_k = u_mat.data[at(i, k)].clone();
                if entry_i_k.is_zero() {
                    continue;
                }
                let factor = Self::exact_div(&entry_i_k, &pivot_val)?;
                l_mat.data[at(i, k)] = factor.clone();
                for j in k..n {
                    let prod = Self::exact_mul(&factor, &u_mat.data[at(k, j)]);
                    u_mat.data[at(i, j)] = Self::exact_sub(&u_mat.data[at(i, j)], &prod);
                }
            }
        }

        let cert = LuCertificate {
            p: p_mat,
            l: l_mat,
            u: u_mat,
        };
        verify_lu_certificate(self, &cert)?;
        Ok(cert)
    }

    /// Solves an exact-rational linear system $A \cdot X = B$ for square $A$ and right-hand side $B$.
    pub fn solve(&self, b: &Matrix) -> Result<Matrix, MatrixError> {
        self.validate_shape()?;
        b.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        if self.rows != b.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, b.rows, b.cols,
            ));
        }
        let n = self.rows;
        let m = b.cols;
        // Build augmented matrix [A | B] of shape n x (n + m)
        let augmented_cols = n.checked_add(m).ok_or_else(|| {
            MatrixError::ResourceLimit("augmented matrix column count overflowed".to_string())
        })?;
        let augmented_entries = Self::checked_element_count(n, augmented_cols)?;
        let mut aug_data = Vec::with_capacity(augmented_entries);
        for r in 0..n {
            for c in 0..n {
                aug_data.push(self.get(r, c)?.clone());
            }
            for c in 0..m {
                aug_data.push(b.get(r, c)?.clone());
            }
        }
        let aug_mat = Matrix::new(n, augmented_cols, aug_data)?;
        let (rref_aug, pivot_cols) = aug_mat.rref()?;
        if pivot_cols.len() < n || pivot_cols.iter().any(|&c| c >= n) {
            return Err(MatrixError::SingularMatrix);
        }
        // Extract solution X of shape n x m from right columns
        let solution_entries = Self::checked_element_count(n, m)?;
        let mut x_data = Vec::with_capacity(solution_entries);
        for r in 0..n {
            for c in 0..m {
                x_data.push(rref_aug.get(r, n + c)?.clone());
            }
        }
        let sol = Matrix::new(n, m, x_data)?;
        let cert = LinearSystemCertificate {
            solution: sol.clone(),
        };
        verify_linear_system_certificate(self, b, &cert)?;
        Ok(sol)
    }

    /// Solves an exact-rational least-squares system $A^T A X = A^T B$.
    pub fn solve_least_squares(&self, b: &Matrix) -> Result<Matrix, MatrixError> {
        let at = self.transpose();
        let ata = at.matmul(self)?;
        let atb = at.matmul(b)?;
        ata.solve(&atb)
    }

    /// Exact-rational QR decomposition with orthogonal columns via Modified Gram-Schmidt: $A = Q \cdot R$.
    ///
    /// For an $m \times n$ matrix $A$ with linearly independent columns ($m \ge n$), computes:
    /// - $Q$ ($m \times n$) whose columns are mutually orthogonal: $Q^T \cdot Q = D$ (diagonal with non-zero entries).
    /// - $R$ ($n \times n$) which is unit upper-triangular with $R_{j, j} = 1$.
    pub fn qr(&self) -> Result<QrCertificate, MatrixError> {
        self.validate_certificate_domain()?;
        let (m, n) = (self.rows, self.cols);
        if m < n {
            return Err(MatrixError::ShapeMismatch(m, n, n, n));
        }

        // Initialize column vectors v_j from A
        let mut v = Vec::with_capacity(n);
        for c in 0..n {
            let mut col_vec = Vec::with_capacity(m);
            for r in 0..m {
                col_vec.push(self.get(r, c)?.clone());
            }
            v.push(col_vec);
        }

        let mut r_data = vec![Expr::from_i64(0); n * n];
        let at_r = |r: usize, c: usize| r * n + c;

        for j in 0..n {
            r_data[at_r(j, j)] = Expr::from_i64(1);
            // Compute norm squared <v_j, v_j>
            let mut norm_sq = Expr::from_i64(0);
            for val in &v[j] {
                let term = Self::exact_mul(val, val);
                norm_sq = Self::exact_add(&norm_sq, &term);
            }
            if norm_sq.is_zero() {
                return Err(MatrixError::SingularMatrix);
            }

            for k in (j + 1)..n {
                // Compute inner product <v_j, A_k>
                let mut dot = Expr::from_i64(0);
                for (i, v_j_i) in v[j].iter().enumerate() {
                    let a_i_k = self.get(i, k)?;
                    let term = Self::exact_mul(v_j_i, a_i_k);
                    dot = Self::exact_add(&dot, &term);
                }
                let proj = Self::exact_div(&dot, &norm_sq)?;
                r_data[at_r(j, k)] = proj.clone();

                // Subtract projection from v_k: v_k -= proj * v_j
                let (left, right) = v.split_at_mut(k);
                for (v_k_i, v_j_i) in right[0].iter_mut().zip(&left[j]) {
                    let sub_term = Self::exact_mul(&proj, v_j_i);
                    *v_k_i = Self::exact_sub(v_k_i, &sub_term);
                }
            }
        }

        // Build Q matrix from columns v_0 .. v_{n-1}
        let mut q_data = Vec::with_capacity(m * n);
        for r in 0..m {
            for col in &v {
                q_data.push(col[r].clone());
            }
        }

        let q_mat = Matrix::new(m, n, q_data)?;
        let r_mat = Matrix::new(n, n, r_data)?;
        let cert = QrCertificate { q: q_mat, r: r_mat };
        verify_qr_certificate(self, &cert)?;
        Ok(cert)
    }

    /// Exact-rational $LDL^T$ decomposition for symmetric matrix $A$: $A = L \cdot D \cdot L^T$.
    ///
    /// Computes unit lower-triangular $L$ and diagonal $D$.
    pub fn ldl(&self) -> Result<LdlCertificate, MatrixError> {
        self.validate_shape()?;
        if self.rows != self.cols {
            return Err(MatrixError::NotSquare(self.rows, self.cols));
        }
        let n = self.rows;
        // Check symmetry: A == A^T
        for i in 0..n {
            for j in (i + 1)..n {
                if self.get(i, j)? != self.get(j, i)? {
                    return Err(MatrixError::ResourceLimit(
                        "LDL decomposition requires a symmetric matrix".to_string(),
                    ));
                }
            }
        }

        let mut l_data = vec![Expr::from_i64(0); n * n];
        let mut d_data = vec![Expr::from_i64(0); n * n];
        let at = |r: usize, c: usize| r * n + c;

        for j in 0..n {
            l_data[at(j, j)] = Expr::from_i64(1);
            // D_jj = A_jj - sum_{k=0}^{j-1} L_jk^2 * D_kk
            let mut sum_d = Expr::from_i64(0);
            for k in 0..j {
                let l_j_k = &l_data[at(j, k)];
                let d_k_k = &d_data[at(k, k)];
                let l_sq = Self::exact_mul(l_j_k, l_j_k);
                let term = Self::exact_mul(&l_sq, d_k_k);
                sum_d = Self::exact_add(&sum_d, &term);
            }
            let a_j_j = self.get(j, j)?;
            let d_j_j = Self::exact_sub(a_j_j, &sum_d);
            if d_j_j.is_zero() {
                return Err(MatrixError::SingularMatrix);
            }
            d_data[at(j, j)] = d_j_j.clone();

            for i in (j + 1)..n {
                // L_ij = (1 / D_jj) * (A_ij - sum_{k=0}^{j-1} L_ik * L_jk * D_kk)
                let mut sum_l = Expr::from_i64(0);
                for k in 0..j {
                    let l_i_k = &l_data[at(i, k)];
                    let l_j_k = &l_data[at(j, k)];
                    let d_k_k = &d_data[at(k, k)];
                    let prod = Self::exact_mul(l_i_k, l_j_k);
                    let term = Self::exact_mul(&prod, d_k_k);
                    sum_l = Self::exact_add(&sum_l, &term);
                }
                let a_i_j = self.get(i, j)?;
                let num = Self::exact_sub(a_i_j, &sum_l);
                let l_i_j = Self::exact_div(&num, &d_j_j)?;
                l_data[at(i, j)] = l_i_j;
            }
        }

        let l_mat = Matrix::new(n, n, l_data)?;
        let d_mat = Matrix::new(n, n, d_data)?;
        let cert = LdlCertificate { l: l_mat, d: d_mat };
        verify_ldl_certificate(self, &cert)?;
        Ok(cert)
    }
}

/// Certificate candidate for an exact-rational linear system solution $A \cdot X = B$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearSystemCertificate {
    pub solution: Matrix,
}

/// Same-crate reference validator checking that exact-rational matrices satisfy $A \cdot X = B$.
pub fn verify_linear_system_certificate(
    a: &Matrix,
    b: &Matrix,
    cert: &LinearSystemCertificate,
) -> Result<(), MatrixError> {
    a.validate_certificate_domain()?;
    b.validate_certificate_domain()?;
    cert.solution.validate_certificate_domain()?;
    if a.cols != cert.solution.rows || a.rows != b.rows || cert.solution.cols != b.cols {
        return Err(MatrixError::ShapeMismatch(
            a.rows,
            a.cols,
            cert.solution.rows,
            cert.solution.cols,
        ));
    }
    let ax = a.matmul(&cert.solution)?;
    if ax != *b {
        return Err(MatrixError::ResourceLimit(
            "Linear system solution verification failed: A * X != B".to_string(),
        ));
    }
    Ok(())
}

/// Certificate candidate for an exact-rational QR decomposition: $A = Q \cdot R$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QrCertificate {
    pub q: Matrix,
    pub r: Matrix,
}

/// Same-crate reference validator checking exact-rational $A = Q \cdot R$, unit upper-triangular $R$, and diagonal $Q^T \cdot Q$.
pub fn verify_qr_certificate(matrix: &Matrix, cert: &QrCertificate) -> Result<(), MatrixError> {
    matrix.validate_certificate_domain()?;
    cert.q.validate_certificate_domain()?;
    cert.r.validate_certificate_domain()?;
    let (m, n) = (matrix.rows, matrix.cols);
    if cert.q.rows != m || cert.q.cols != n || cert.r.rows != n || cert.r.cols != n {
        return Err(MatrixError::ShapeMismatch(m, n, cert.q.rows, cert.q.cols));
    }
    // Check R is upper-triangular with unit diagonal
    for i in 0..n {
        for j in 0..n {
            let entry = cert.r.get(i, j)?;
            if i == j {
                if !entry.is_one() {
                    return Err(MatrixError::ResourceLimit(
                        "R diagonal entries must be 1 in unit upper-triangular QR".to_string(),
                    ));
                }
            } else if i > j && !entry.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "R lower triangular entries must be 0".to_string(),
                ));
            }
        }
    }
    // Check Q^T * Q is diagonal (columns of Q are mutually orthogonal)
    let qt = cert.q.transpose();
    let qt_q = qt.matmul(&cert.q)?;
    for i in 0..n {
        for j in 0..n {
            let entry = qt_q.get(i, j)?;
            if i != j && !entry.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "Q column vectors are not mutually orthogonal (Q^T * Q is not diagonal)"
                        .to_string(),
                ));
            }
            if i == j && entry.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "Q column vector norm is zero".to_string(),
                ));
            }
        }
    }
    // Check Q * R == A
    let qr = cert.q.matmul(&cert.r)?;
    if qr != *matrix {
        return Err(MatrixError::ResourceLimit(
            "QR factorization check failed: Q * R != A".to_string(),
        ));
    }
    Ok(())
}

/// Certificate candidate for an exact-rational $LDL^T$ decomposition: $A = L \cdot D \cdot L^T$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LdlCertificate {
    pub l: Matrix,
    pub d: Matrix,
}

/// Same-crate reference validator checking exact-rational $A = L \cdot D \cdot L^T$, unit lower-triangular $L$, and diagonal $D$.
pub fn verify_ldl_certificate(matrix: &Matrix, cert: &LdlCertificate) -> Result<(), MatrixError> {
    matrix.validate_certificate_domain()?;
    cert.l.validate_certificate_domain()?;
    cert.d.validate_certificate_domain()?;
    if matrix.rows != matrix.cols {
        return Err(MatrixError::NotSquare(matrix.rows, matrix.cols));
    }
    let n = matrix.rows;
    if cert.l.rows != n || cert.l.cols != n || cert.d.rows != n || cert.d.cols != n {
        return Err(MatrixError::ShapeMismatch(n, n, cert.l.rows, cert.l.cols));
    }
    // Check L is unit lower-triangular
    for i in 0..n {
        for j in 0..n {
            let entry = cert.l.get(i, j)?;
            if i == j {
                if !entry.is_one() {
                    return Err(MatrixError::ResourceLimit(
                        "L diagonal entries must be 1 in LDL decomposition".to_string(),
                    ));
                }
            } else if j > i && !entry.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "L upper-triangular entries must be 0".to_string(),
                ));
            }
        }
    }
    // Check D is diagonal
    for i in 0..n {
        for j in 0..n {
            if i != j && !cert.d.get(i, j)?.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "D off-diagonal entries must be 0".to_string(),
                ));
            }
        }
    }
    // Check L * D * L^T == A
    let ld = cert.l.matmul(&cert.d)?;
    let lt = cert.l.transpose();
    let ldlt = ld.matmul(&lt)?;
    if ldlt != *matrix {
        return Err(MatrixError::ResourceLimit(
            "LDL factorization check failed: L * D * L^T != A".to_string(),
        ));
    }
    Ok(())
}

/// Certificate candidate for an exact-rational matrix inverse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseCertificate {
    pub inverse: Matrix,
}

/// Same-crate reference validator checking an exact-rational inverse on both multiplication sides.
pub fn verify_inverse_certificate(
    matrix: &Matrix,
    cert: &InverseCertificate,
) -> Result<(), MatrixError> {
    matrix.validate_certificate_domain()?;
    cert.inverse.validate_certificate_domain()?;
    if matrix.rows != matrix.cols
        || cert.inverse.rows != cert.inverse.cols
        || matrix.rows != cert.inverse.rows
    {
        return Err(MatrixError::ShapeMismatch(
            matrix.rows,
            matrix.cols,
            cert.inverse.rows,
            cert.inverse.cols,
        ));
    }
    let n = matrix.rows;
    let eye = Matrix::eye(n)?;
    let prod1 = matrix.matmul(&cert.inverse)?;
    let prod2 = cert.inverse.matmul(matrix)?;
    if prod1 != eye || prod2 != eye {
        return Err(MatrixError::ResourceLimit(
            "Inverse certificate verification failed: A * A^-1 != I".to_string(),
        ));
    }
    Ok(())
}

/// Certificate candidate for an exact-rational matrix nullspace basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullspaceCertificate {
    pub basis: Vec<Matrix>,
}

#[derive(Serialize)]
struct NullspaceCertificateWireRef<'a> {
    basis: &'a [Matrix],
}

impl Serialize for NullspaceCertificate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.basis.len() > MAX_NULLSPACE_BASIS_VECTORS {
            return Err(serde::ser::Error::custom(format!(
                "nullspace basis exceeds the vector limit of {MAX_NULLSPACE_BASIS_VECTORS}"
            )));
        }
        let aggregate_entries = self.basis.iter().try_fold(0usize, |entries, vector| {
            entries
                .checked_add(vector.data.len())
                .ok_or_else(|| serde::ser::Error::custom("nullspace basis entry count overflowed"))
        })?;
        if aggregate_entries > MAX_NULLSPACE_BASIS_ENTRIES {
            return Err(serde::ser::Error::custom(format!(
                "nullspace basis exceeds the aggregate entry limit of {MAX_NULLSPACE_BASIS_ENTRIES}"
            )));
        }
        NullspaceCertificateWireRef { basis: &self.basis }.serialize(serializer)
    }
}

struct BoundedMatrixVec(Vec<Matrix>);

impl<'de> Deserialize<'de> for BoundedMatrixVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BoundedMatrixVecVisitor;

        impl<'de> Visitor<'de> for BoundedMatrixVecVisitor {
            type Value = BoundedMatrixVec;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "a nullspace basis with at most {MAX_NULLSPACE_BASIS_VECTORS} vectors and {MAX_NULLSPACE_BASIS_ENTRIES} aggregate entries"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let hinted = sequence.size_hint().unwrap_or(0);
                if hinted > MAX_NULLSPACE_BASIS_VECTORS {
                    return Err(serde::de::Error::custom(format!(
                        "nullspace basis exceeds the vector limit of {MAX_NULLSPACE_BASIS_VECTORS}"
                    )));
                }
                let mut basis = Vec::with_capacity(hinted.min(MAX_NULLSPACE_BASIS_VECTORS));
                let mut aggregate_entries = 0usize;
                loop {
                    if basis.len() == MAX_NULLSPACE_BASIS_VECTORS {
                        if sequence.next_element::<IgnoredAny>()?.is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "nullspace basis exceeds the vector limit of {MAX_NULLSPACE_BASIS_VECTORS}"
                            )));
                        }
                        break;
                    }
                    let Some(vector) = sequence.next_element::<Matrix>()? else {
                        break;
                    };
                    aggregate_entries = aggregate_entries
                        .checked_add(vector.data.len())
                        .ok_or_else(|| {
                            serde::de::Error::custom("nullspace basis entry count overflowed")
                        })?;
                    if aggregate_entries > MAX_NULLSPACE_BASIS_ENTRIES {
                        return Err(serde::de::Error::custom(format!(
                            "nullspace basis exceeds the aggregate entry limit of {MAX_NULLSPACE_BASIS_ENTRIES}"
                        )));
                    }
                    basis.push(vector);
                }
                Ok(BoundedMatrixVec(basis))
            }
        }

        deserializer.deserialize_seq(BoundedMatrixVecVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NullspaceCertificateWire {
    basis: BoundedMatrixVec,
}

impl<'de> Deserialize<'de> for NullspaceCertificate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = NullspaceCertificateWire::deserialize(deserializer)?;
        Ok(Self {
            basis: wire.basis.0,
        })
    }
}

/// Same-crate reference validator checking an exact-rational nullspace basis and rank-nullity.
pub fn verify_nullspace_certificate(
    matrix: &Matrix,
    cert: &NullspaceCertificate,
) -> Result<(), MatrixError> {
    matrix.validate_certificate_domain()?;
    let rank = matrix.rank()?;
    let expected_nullity = matrix.cols.saturating_sub(rank);
    check_nullspace_basis_size(matrix.cols, expected_nullity)?;
    if cert.basis.len() != expected_nullity {
        return Err(MatrixError::ResourceLimit(format!(
            "Nullspace certificate basis size {} does not match rank-nullity expected nullity {}",
            cert.basis.len(),
            expected_nullity
        )));
    }
    for (idx, v) in cert.basis.iter().enumerate() {
        v.validate_certificate_domain()?;
        if v.rows != matrix.cols || v.cols != 1 {
            return Err(MatrixError::ShapeMismatch(
                matrix.rows,
                matrix.cols,
                v.rows,
                v.cols,
            ));
        }
        let av = matrix.matmul(v)?;
        for r in 0..av.rows {
            if !av.get(r, 0)?.is_zero() {
                return Err(MatrixError::ResourceLimit(format!(
                    "Nullspace basis vector {idx} does not satisfy A * v = 0"
                )));
            }
        }
    }
    if !cert.basis.is_empty() {
        let mut stacked_data = Vec::with_capacity(matrix.cols * cert.basis.len());
        for r in 0..matrix.cols {
            for v in &cert.basis {
                stacked_data.push(v.get(r, 0)?.clone());
            }
        }
        let stacked = Matrix::new(matrix.cols, cert.basis.len(), stacked_data)?;
        if stacked.rank()? != cert.basis.len() {
            return Err(MatrixError::ResourceLimit(
                "Nullspace basis vectors are not linearly independent".to_string(),
            ));
        }
    }
    Ok(())
}

/// Certificate candidate for exact-rational LU with partial row pivoting: $P \cdot A = L \cdot U$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LuCertificate {
    pub p: Matrix,
    pub l: Matrix,
    pub u: Matrix,
}

/// Same-crate reference validator checking exact-rational $P \cdot A = L \cdot U$, a 0/1 permutation $P$, unit lower-triangular $L$, and upper-triangular $U$.
pub fn verify_lu_certificate(matrix: &Matrix, cert: &LuCertificate) -> Result<(), MatrixError> {
    matrix.validate_certificate_domain()?;
    cert.p.validate_certificate_domain()?;
    cert.l.validate_certificate_domain()?;
    cert.u.validate_certificate_domain()?;
    if matrix.rows != matrix.cols {
        return Err(MatrixError::NotSquare(matrix.rows, matrix.cols));
    }
    let n = matrix.rows;
    if cert.p.rows != n
        || cert.p.cols != n
        || cert.l.rows != n
        || cert.l.cols != n
        || cert.u.rows != n
        || cert.u.cols != n
    {
        return Err(MatrixError::ShapeMismatch(n, n, cert.p.rows, cert.p.cols));
    }
    let mut column_ones = vec![0usize; n];
    for row in 0..n {
        let mut row_ones = 0usize;
        for (column, column_count) in column_ones.iter_mut().enumerate() {
            let entry = cert.p.get(row, column)?;
            if entry.is_zero() {
                continue;
            }
            if entry.is_one() {
                row_ones += 1;
                *column_count += 1;
            } else {
                return Err(MatrixError::InvalidPermutationMatrix);
            }
        }
        if row_ones != 1 {
            return Err(MatrixError::InvalidPermutationMatrix);
        }
    }
    if column_ones.iter().any(|&ones| ones != 1) {
        return Err(MatrixError::InvalidPermutationMatrix);
    }

    for i in 0..n {
        for j in 0..n {
            let entry = cert.l.get(i, j)?;
            if i == j {
                if !entry.is_one() {
                    return Err(MatrixError::ResourceLimit(
                        "L diagonal entries must be 1".to_string(),
                    ));
                }
            } else if j > i && !entry.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "L upper triangular entries must be 0".to_string(),
                ));
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            if i > j && !cert.u.get(i, j)?.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "U lower triangular entries must be 0".to_string(),
                ));
            }
        }
    }
    let pa = cert.p.matmul(matrix)?;
    let lu = cert.l.matmul(&cert.u)?;
    if pa != lu {
        return Err(MatrixError::ResourceLimit(
            "LU factorization check failed: P * A != L * U".to_string(),
        ));
    }
    Ok(())
}

/// Computes `det(point * I - matrix)` through exact rational Gaussian elimination.
///
/// This deliberately does not call the Faddeev-LeVerrier characteristic-polynomial
/// generator. It is the bounded reference lane used by the certificate validator.
fn reference_shifted_determinant(
    matrix: &Matrix,
    point: &BigRational,
) -> Result<BigRational, MatrixError> {
    let n = matrix.rows;
    if n == 0 {
        return Ok(BigRational::one());
    }

    let entry_count = n.checked_mul(n).ok_or(MatrixError::ShapeOverflow(n, n))?;
    let mut work = Vec::new();
    work.try_reserve_exact(entry_count).map_err(|_| {
        MatrixError::ResourceLimit(format!(
            "reference determinant could not reserve {entry_count} rational entries"
        ))
    })?;
    for row in 0..n {
        for col in 0..n {
            let entry = Matrix::numeric(&matrix.data[row * n + col])
                .ok_or(MatrixError::UnsupportedCertificateDomain)?;
            work.push(if row == col {
                point.clone() - entry
            } else {
                -entry
            });
        }
    }

    let mut determinant = BigRational::one();
    for pivot_col in 0..n {
        let Some(pivot_row) = (pivot_col..n).find(|&row| !work[row * n + pivot_col].is_zero())
        else {
            return Ok(BigRational::zero());
        };

        if pivot_row != pivot_col {
            for col in 0..n {
                work.swap(pivot_row * n + col, pivot_col * n + col);
            }
            determinant = -determinant;
        }

        let pivot = work[pivot_col * n + pivot_col].clone();
        determinant *= pivot.clone();
        for row in (pivot_col + 1)..n {
            let row_offset = row * n;
            let leading = work[row_offset + pivot_col].clone();
            if leading.is_zero() {
                continue;
            }
            let factor = leading / pivot.clone();
            work[row_offset + pivot_col] = BigRational::zero();
            let pivot_offset = pivot_col * n;
            for col in (pivot_col + 1)..n {
                let correction = factor.clone() * work[pivot_offset + col].clone();
                work[row_offset + col] -= correction;
            }
        }
    }
    Ok(determinant)
}

/// Certificate candidate for an exact-rational matrix characteristic polynomial $P(\lambda) = \det(\lambda I - A)$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharpolyCertificate {
    pub poly: UnivariatePoly,
}

/// Same-crate reference validator checking an exact-rational characteristic polynomial.
///
/// Acceptance criteria:
/// 1. Matrix is square $n \times n$ with exact rational entries.
/// 2. Polynomial degree equals $n$ and is monic ($\text{LC}(P) = 1$).
/// 3. At each of `n + 1` distinct exact points, `P(t) = det(t I - A)` according
///    to a rational-elimination reference lane independent of the generator.
///
/// Since both sides have degree at most `n`, agreement at `n + 1` distinct
/// points establishes equality. Cayley-Hamilton plus trace/determinant anchors
/// is not sufficient for matrices whose minimal polynomial has degree below `n`.
pub fn verify_charpoly_certificate(
    matrix: &Matrix,
    cert: &CharpolyCertificate,
) -> Result<(), MatrixError> {
    matrix.validate_shape()?;
    if matrix.rows != matrix.cols {
        return Err(MatrixError::NotSquare(matrix.rows, matrix.cols));
    }
    let n = matrix.rows;
    if n > MAX_CHARACTERISTIC_POLYNOMIAL_DIMENSION {
        return Err(MatrixError::ResourceLimit(format!(
            "characteristic polynomial verification supports dimensions up to {MAX_CHARACTERISTIC_POLYNOMIAL_DIMENSION}"
        )));
    }
    if matrix
        .data
        .iter()
        .any(|entry| !matches!(entry, Expr::Integer(_) | Expr::Rational(_)))
    {
        return Err(MatrixError::UnsupportedCertificateDomain);
    }
    cert.poly
        .validate_shape()
        .map_err(|error| MatrixError::InvalidCertificate(error.to_string()))?;
    if cert.poly.degree() != Some(n) {
        return Err(MatrixError::InvalidCertificate(format!(
            "characteristic polynomial degree {:?} does not match matrix dimension {}",
            cert.poly.degree(),
            n
        )));
    }
    if !cert.poly.is_monic() {
        return Err(MatrixError::InvalidCertificate(
            "characteristic polynomial must be monic".to_string(),
        ));
    }

    let sample_count = n
        .checked_add(1)
        .ok_or_else(|| MatrixError::ResourceLimit("sample count overflowed".to_string()))?;
    for sample in 0..sample_count {
        let sample_i64 = i64::try_from(sample).map_err(|_| {
            MatrixError::ResourceLimit("sample point exceeds the supported i64 range".to_string())
        })?;
        let point = BigRational::from_integer(BigInt::from(sample_i64));
        let claimed = cert.poly.eval(&point);
        let expected = reference_shifted_determinant(matrix, &point)?;
        if claimed != expected {
            return Err(MatrixError::InvalidCertificate(format!(
                "characteristic polynomial mismatch at reference sample {sample}"
            )));
        }
    }
    Ok(())
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
        let eye = Matrix::eye(3).unwrap();
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
    fn exact_fold_handles_typed_nonnegative_rational_powers() {
        let rational_power = Expr::Pow(
            std::sync::Arc::new(Expr::Rational(BigRational::new(
                BigInt::from(2),
                BigInt::from(3),
            ))),
            std::sync::Arc::new(Expr::Integer(BigInt::from(3))),
        );
        assert_eq!(
            exact_fold(&rational_power),
            Expr::Rational(BigRational::new(BigInt::from(8), BigInt::from(27)))
        );

        let zero_to_zero = Expr::Pow(
            std::sync::Arc::new(Expr::Rational(BigRational::zero())),
            std::sync::Arc::new(Expr::Integer(BigInt::zero())),
        );
        assert_eq!(exact_fold(&zero_to_zero), Expr::Integer(BigInt::one()));

        let reciprocal = Expr::Pow(
            std::sync::Arc::new(Expr::Rational(BigRational::new(
                BigInt::from(2),
                BigInt::from(3),
            ))),
            std::sync::Arc::new(Expr::Integer(BigInt::from(-1))),
        );
        assert_eq!(
            exact_fold(&reciprocal),
            Expr::Rational(BigRational::new(BigInt::from(3), BigInt::from(2)))
        );

        let zero_reciprocal = Expr::Pow(
            std::sync::Arc::new(Expr::Rational(BigRational::zero())),
            std::sync::Arc::new(Expr::Integer(BigInt::from(-1))),
        );
        let folded = std::panic::catch_unwind(|| exact_fold(&zero_reciprocal));
        assert!(folded.is_ok(), "exact folding zero reciprocal unwound");
        if let Ok(folded) = folded {
            assert_eq!(
                folded,
                Expr::Pow(
                    std::sync::Arc::new(Expr::Integer(BigInt::zero())),
                    std::sync::Arc::new(Expr::Integer(BigInt::from(-1))),
                )
            );
        }
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
        assert_eq!(Matrix::eye(4).unwrap().rank().unwrap(), 4);
        assert_eq!(Matrix::zeros(2, 3).unwrap().rank().unwrap(), 0);

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
        assert_eq!(m.rank().unwrap(), 2);
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
            assert!(
                matches!(root, Expr::Mul(parts) if parts.len() == 2),
                "expected two-factor quadratic-formula expression, got {root:?}"
            );
        }
    }

    #[test]
    fn eigenvalues_beyond_quadratic_degree_refused() {
        let m = Matrix::eye(3).unwrap();
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

    #[test]
    fn test_rref_and_nullspace_computation() {
        // A = [ [1, 2, 1],
        //       [2, 4, 2],
        //       [3, 6, 4] ]
        let m = Matrix::new(
            3,
            3,
            vec![
                num(1),
                num(2),
                num(1),
                num(2),
                num(4),
                num(2),
                num(3),
                num(6),
                num(4),
            ],
        )
        .unwrap();

        let (rref_mat, pivots) = m.rref().unwrap();
        assert_eq!(pivots, vec![0, 2]); // Pivots at col 0 and col 2
        // RREF should be:
        // [ [1, 2, 0],
        //   [0, 0, 1],
        //   [0, 0, 0] ]
        assert_eq!(rref_mat.get(0, 0).unwrap(), &num(1));
        assert_eq!(rref_mat.get(0, 1).unwrap(), &num(2));
        assert_eq!(rref_mat.get(0, 2).unwrap(), &num(0));
        assert_eq!(rref_mat.get(1, 1).unwrap(), &num(0));
        assert_eq!(rref_mat.get(1, 2).unwrap(), &num(1));

        let ns = m.nullspace().unwrap();
        assert_eq!(ns.len(), 1); // 1 free variable (col 1)
        // Nullspace vector: [-2, 1, 0]^T
        let v = &ns[0];
        assert_eq!(v.get(0, 0).unwrap(), &num(-2));
        assert_eq!(v.get(1, 0).unwrap(), &num(1));
        assert_eq!(v.get(2, 0).unwrap(), &num(0));

        // Check A * v = 0
        let av = m.matmul(v).unwrap();
        for r in 0..3 {
            assert!(av.get(r, 0).unwrap().is_zero());
        }
    }

    #[test]
    fn test_jacobian_computation() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let vars = vec![x.clone(), y.clone()];

        // f1 = x^2 * y
        // f2 = x + 3*y
        let f1 = Expr::Mul(vec![
            Expr::Pow(
                std::sync::Arc::new(Expr::Sym(x.clone())),
                std::sync::Arc::new(Expr::from_i64(2)),
            ),
            Expr::Sym(y.clone()),
        ]);
        let f2 = Expr::Add(vec![
            Expr::Sym(x.clone()),
            Expr::Mul(vec![Expr::from_i64(3), Expr::Sym(y.clone())]),
        ]);

        let j = Matrix::jacobian(&[f1, f2], &vars).unwrap();
        assert_eq!(j.rows, 2);
        assert_eq!(j.cols, 2);

        // J[0, 0] = 2*x*y
        // J[0, 1] = x^2
        // J[1, 0] = 1
        // J[1, 1] = 3
        assert_eq!(j.get(1, 0).unwrap(), &num(1));
        assert_eq!(j.get(1, 1).unwrap(), &num(3));
    }

    #[test]
    fn test_sparse_dense_roundtrip_and_matmul() {
        let m = Matrix::new(
            3,
            3,
            vec![
                num(0),
                num(2),
                num(0),
                num(0),
                num(0),
                num(3),
                num(4),
                num(0),
                num(0),
            ],
        )
        .unwrap();

        let s = m.to_sparse().unwrap();
        assert_eq!(s.entries().len(), 3);
        let m_roundtrip = s.to_dense().unwrap();
        assert_eq!(m, m_roundtrip);

        let s_eye = SparseMatrix::eye(3).unwrap();
        let s_prod = s.matmul(&s_eye).unwrap();
        assert_eq!(s_prod.to_dense().unwrap(), m);
    }

    #[test]
    fn matrix_construction_and_wire_decode_reject_invalid_shapes() {
        assert_eq!(
            Matrix::new(usize::MAX, 2, Vec::new()).unwrap_err(),
            MatrixError::ShapeOverflow(usize::MAX, 2)
        );
        assert!(matches!(
            Matrix::new(MAX_MATRIX_ENTRIES + 1, 1, Vec::new()),
            Err(MatrixError::EntryLimitExceeded(_))
        ));

        let valid = Matrix::new(1, 1, vec![num(1)]).unwrap();
        let mut wire = serde_json::to_value(valid).unwrap();
        wire["rows"] = serde_json::json!(2);
        wire["cols"] = serde_json::json!(2);
        assert!(serde_json::from_value::<Matrix>(wire).is_err());
    }

    #[test]
    fn empty_and_scalar_matrix_algebra_is_exact() {
        let empty = Matrix::new(0, 0, Vec::new()).unwrap();
        assert_eq!(empty.det().unwrap(), num(1));
        assert!(empty.eigenvalues().unwrap().is_empty());

        let scalar = Matrix::new(1, 1, vec![num(4)]).unwrap();
        assert_eq!(
            scalar.inverse().unwrap().get(0, 0).unwrap(),
            &Expr::Rational(BigRational::new(1.into(), 4.into()))
        );
    }

    #[test]
    fn symbolic_zero_decisions_are_refused() {
        let symbolic = Matrix::new(1, 1, vec![Expr::symbol("x")]).unwrap();
        assert_eq!(
            symbolic.inverse().unwrap_err(),
            MatrixError::SymbolicZeroUndetermined
        );
        assert_eq!(
            symbolic.rank().unwrap_err(),
            MatrixError::SymbolicZeroUndetermined
        );
        assert_eq!(
            symbolic.rref().unwrap_err(),
            MatrixError::SymbolicZeroUndetermined
        );
    }

    #[test]
    fn sparse_wire_rejects_out_of_bounds_entries() {
        let mut entries = std::collections::BTreeMap::new();
        entries.insert((0, 0), num(1));
        let sparse = SparseMatrix::new(1, 1, entries).unwrap();
        let mut wire = serde_json::to_value(sparse).unwrap();
        wire["entries"][0]["row"] = serde_json::json!(1);
        assert!(serde_json::from_value::<SparseMatrix>(wire).is_err());
    }

    #[test]
    fn test_lu_decomposition_and_verification() {
        // A = [[2, 4], [1, 7]]
        let a = Matrix::new(2, 2, vec![num(2), num(4), num(1), num(7)]).unwrap();
        let lu_cert = a.lu().unwrap();
        assert!(verify_lu_certificate(&a, &lu_cert).is_ok());

        // A = [[0, 3], [2, 1]] (requires row swap pivot)
        let a_swap = Matrix::new(2, 2, vec![num(0), num(3), num(2), num(1)]).unwrap();
        let lu_swap = a_swap.lu().unwrap();
        assert!(verify_lu_certificate(&a_swap, &lu_swap).is_ok());

        // 3x3 matrix
        let a_3x3 = Matrix::new(
            3,
            3,
            vec![
                num(1),
                num(2),
                num(3),
                num(2),
                num(5),
                num(7),
                num(3),
                num(1),
                num(2),
            ],
        )
        .unwrap();
        let lu_3x3 = a_3x3.lu().unwrap();
        assert!(verify_lu_certificate(&a_3x3, &lu_3x3).is_ok());
    }

    #[test]
    fn test_inverse_certificate_and_verification() {
        let m = Matrix::new(2, 2, vec![num(4), num(7), num(2), num(6)]).unwrap();
        let inv = m.inverse().unwrap();
        let cert = InverseCertificate { inverse: inv };
        assert!(verify_inverse_certificate(&m, &cert).is_ok());
    }

    #[test]
    fn test_nullspace_certificate_and_verification() {
        let m = Matrix::new(
            3,
            3,
            vec![
                num(1),
                num(2),
                num(1),
                num(2),
                num(4),
                num(2),
                num(3),
                num(6),
                num(4),
            ],
        )
        .unwrap();
        let basis = m.nullspace().unwrap();
        let cert = NullspaceCertificate { basis };
        assert!(verify_nullspace_certificate(&m, &cert).is_ok());
    }

    #[test]
    fn test_mutant_tampered_certificates_rejected() {
        let a = Matrix::new(2, 2, vec![num(2), num(4), num(1), num(7)]).unwrap();
        let mut lu_cert = a.lu().unwrap();

        // 1. Tamper L entry
        lu_cert.l.data[1] = num(99);
        assert!(verify_lu_certificate(&a, &lu_cert).is_err());

        // 2. Tamper U entry
        let mut lu_cert2 = a.lu().unwrap();
        lu_cert2.u.data[3] = num(99);
        assert!(verify_lu_certificate(&a, &lu_cert2).is_err());

        // 3. Tamper inverse
        let inv = a.inverse().unwrap();
        let mut inv_cert = InverseCertificate { inverse: inv };
        inv_cert.inverse.data[0] = num(99);
        assert!(verify_inverse_certificate(&a, &inv_cert).is_err());

        // 4. Tamper nullspace vector
        let m = Matrix::new(
            3,
            3,
            vec![
                num(1),
                num(2),
                num(1),
                num(2),
                num(4),
                num(2),
                num(3),
                num(6),
                num(4),
            ],
        )
        .unwrap();
        let mut basis = m.nullspace().unwrap();
        basis[0].data[0] = num(99);
        let ns_cert = NullspaceCertificate { basis };
        assert!(verify_nullspace_certificate(&m, &ns_cert).is_err());
    }

    #[test]
    fn zero_dimension_metered_operations_do_not_issue_zero_charges() {
        let empty = Matrix::new(0, 0, Vec::new()).unwrap();
        let mut budget = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(1, 0));
        assert_eq!(empty.metered_matmul(&empty, &mut budget).unwrap(), empty);
        let (reduced, pivots) = empty.metered_rref(&mut budget).unwrap();
        assert_eq!(reduced, empty);
        assert!(pivots.is_empty());
    }

    #[test]
    fn test_exact_linear_system_solve_and_verification() {
        // [ [2, 1], [5, 7] ] * [x, y]^T = [11, 28]^T => x = 7, y = -3 (2*7 - 3 = 11, 5*7 - 21 = 14 != 28 -> let's compute exact: 2*x + y = 11, 5*x + 7*y = 28 -> det = 14-5=9. x = (77-28)/9 = 49/9, y = (56-55)/9 = 1/9)
        let a = Matrix::new(2, 2, vec![num(2), num(1), num(5), num(7)]).unwrap();
        let b = Matrix::new(2, 1, vec![num(11), num(28)]).unwrap();
        let x = a.solve(&b).unwrap();
        assert_eq!(
            *x.get(0, 0).unwrap(),
            Expr::Rational(BigRational::new(49.into(), 9.into()))
        );
        assert_eq!(
            *x.get(1, 0).unwrap(),
            Expr::Rational(BigRational::new(1.into(), 9.into()))
        );
        assert_eq!(a.matmul(&x).unwrap(), b);

        // Singular matrix fails
        let sing = Matrix::new(2, 2, vec![num(1), num(2), num(2), num(4)]).unwrap();
        assert_eq!(sing.solve(&b).unwrap_err(), MatrixError::SingularMatrix);
    }

    #[test]
    fn test_exact_least_squares_solve() {
        // Overdetermined system: A is 3x2, b is 3x1
        // A = [ [1, 1], [1, 2], [1, 3] ], b = [ [1], [2], [2] ]
        let a = Matrix::new(3, 2, vec![num(1), num(1), num(1), num(2), num(1), num(3)]).unwrap();
        let b = Matrix::new(3, 1, vec![num(1), num(2), num(2)]).unwrap();
        let x = a.solve_least_squares(&b).unwrap();
        // A^T A = [ [3, 6], [6, 14] ], A^T b = [ [5], [11] ]
        // det = 42 - 36 = 6.
        // x_0 = (70 - 66)/6 = 4/6 = 2/3, x_1 = (33 - 30)/6 = 3/6 = 1/2.
        assert_eq!(
            *x.get(0, 0).unwrap(),
            Expr::Rational(BigRational::new(2.into(), 3.into()))
        );
        assert_eq!(
            *x.get(1, 0).unwrap(),
            Expr::Rational(BigRational::new(1.into(), 2.into()))
        );
    }

    #[test]
    fn test_exact_qr_decomposition_and_verification() {
        let a = Matrix::new(3, 2, vec![num(1), num(2), num(0), num(1), num(1), num(0)]).unwrap();
        let qr_cert = a.qr().unwrap();
        verify_qr_certificate(&a, &qr_cert).unwrap();

        // Mutants:
        // 1. Tamper Q entry
        let mut tampered_q = qr_cert.clone();
        tampered_q.q.data[0] = num(99);
        assert!(verify_qr_certificate(&a, &tampered_q).is_err());

        // 2. Tamper R entry
        let mut tampered_r = qr_cert;
        tampered_r.r.data[0] = num(2); // must be 1 on unit diagonal
        assert!(verify_qr_certificate(&a, &tampered_r).is_err());
    }

    #[test]
    fn test_exact_ldl_decomposition_and_verification() {
        // Symmetric positive definite matrix:
        // [ [4, 12, -16], [12, 37, -43], [-16, -43, 98] ]
        let a = Matrix::new(
            3,
            3,
            vec![
                num(4),
                num(12),
                num(-16),
                num(12),
                num(37),
                num(-43),
                num(-16),
                num(-43),
                num(98),
            ],
        )
        .unwrap();
        let ldl_cert = a.ldl().unwrap();
        verify_ldl_certificate(&a, &ldl_cert).unwrap();

        // D should have diagonal [4, 1, 9]
        assert_eq!(*ldl_cert.d.get(0, 0).unwrap(), num(4));
        assert_eq!(*ldl_cert.d.get(1, 1).unwrap(), num(1));
        assert_eq!(*ldl_cert.d.get(2, 2).unwrap(), num(9));

        // Mutants:
        // 1. Tamper L
        let mut tampered_l = ldl_cert.clone();
        tampered_l.l.data[1] = num(99);
        assert!(verify_ldl_certificate(&a, &tampered_l).is_err());

        // 2. Tamper D off-diagonal
        let mut tampered_d = ldl_cert;
        tampered_d.d.data[1] = num(1);
        assert!(verify_ldl_certificate(&a, &tampered_d).is_err());
    }

    #[test]
    fn zero_area_shapes_still_bound_each_dimension() {
        let oversized = MAX_MATRIX_DIMENSION + 1;

        assert_eq!(
            Matrix::new(0, oversized, Vec::new()),
            Err(MatrixError::DimensionLimitExceeded(0, oversized))
        );
        assert_eq!(
            Matrix::new(oversized, 0, Vec::new()),
            Err(MatrixError::DimensionLimitExceeded(oversized, 0))
        );
    }

    #[test]
    fn nullspace_refuses_quadratic_output_amplification() {
        let matrix = Matrix::zeros(0, MAX_NULLSPACE_BASIS_VECTORS + 1).unwrap();
        assert!(matches!(
            matrix.nullspace(),
            Err(MatrixError::ResourceLimit(message))
                if message.contains("nullspace basis exceeds")
        ));
    }

    #[test]
    fn nullspace_certificate_wire_bounds_vector_count() {
        let empty = Matrix::zeros(0, 0).unwrap();
        let encoded_empty = serde_json::to_value(empty).unwrap();
        let wire = serde_json::to_vec(&serde_json::json!({
            "basis": vec![encoded_empty; MAX_NULLSPACE_BASIS_VECTORS + 1],
        }))
        .unwrap();

        assert!(serde_json::from_slice::<NullspaceCertificate>(&wire).is_err());
    }

    #[test]
    fn nullspace_certificate_wire_refuses_invalid_export() {
        let certificate = NullspaceCertificate {
            basis: vec![Matrix::zeros(0, 0).unwrap(); MAX_NULLSPACE_BASIS_VECTORS + 1],
        };

        assert!(serde_json::to_vec(&certificate).is_err());
    }

    #[test]
    fn qr_refuses_symbolic_nonzero_assumptions() {
        let x = Expr::symbol("x");
        let matrix = Matrix::new(1, 1, vec![x.clone()]).unwrap();
        assert_eq!(matrix.qr(), Err(MatrixError::UnsupportedCertificateDomain));

        let forged = QrCertificate {
            q: Matrix::new(1, 1, vec![x]).unwrap(),
            r: Matrix::eye(1).unwrap(),
        };
        assert_eq!(
            verify_qr_certificate(&matrix, &forged),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }

    #[test]
    fn lu_verifier_rejects_orthogonal_non_permutation_matrix() {
        let rat = |numerator: i64, denominator: i64| {
            Expr::Rational(BigRational::new(numerator.into(), denominator.into()))
        };
        let matrix = Matrix::eye(2).unwrap();
        let p = Matrix::new(2, 2, vec![rat(3, 5), rat(4, 5), rat(-4, 5), rat(3, 5)]).unwrap();
        let l = Matrix::new(2, 2, vec![num(1), num(0), rat(-4, 3), num(1)]).unwrap();
        let u = Matrix::new(2, 2, vec![rat(3, 5), rat(4, 5), num(0), rat(5, 3)]).unwrap();

        // This certificate satisfies the old verifier's orthogonality and decomposition
        // checks, but P is a rotation rather than a row permutation.
        assert_eq!(p.matmul(&p.transpose()).unwrap(), matrix);
        assert_eq!(l.matmul(&u).unwrap(), p);
        let forged = LuCertificate { p, l, u };
        assert_eq!(
            verify_lu_certificate(&matrix, &forged),
            Err(MatrixError::InvalidPermutationMatrix)
        );
    }

    #[test]
    fn charpoly_certificate_verifies_reference_evaluations() {
        let matrix = Matrix::new(2, 2, vec![num(1), num(2), num(3), num(4)]).unwrap();

        let cert = matrix.char_poly_with_certificate("lambda").unwrap();
        assert_eq!(cert.poly.degree(), Some(2));
        assert!(cert.poly.is_monic());
        assert_eq!(cert.poly.coeffs[2], BigRational::one());
        assert_eq!(cert.poly.coeffs[1], BigRational::from_integer((-5).into())); // -trace
        assert_eq!(cert.poly.coeffs[0], BigRational::from_integer((-2).into())); // det = 1*4 - 2*3 = -2

        assert!(verify_charpoly_certificate(&matrix, &cert).is_ok());

        // 3x3 matrix test
        let m3 = Matrix::new(
            3,
            3,
            vec![
                num(2),
                num(-1),
                num(0),
                num(-1),
                num(2),
                num(-1),
                num(0),
                num(-1),
                num(2),
            ],
        )
        .unwrap();
        let cert3 = m3.char_poly_with_certificate("lambda").unwrap();
        assert_eq!(cert3.poly.degree(), Some(3));
        assert!(verify_charpoly_certificate(&m3, &cert3).is_ok());

        // 1x1 matrix test
        let m1 = Matrix::new(1, 1, vec![num(7)]).unwrap();
        let cert1 = m1.char_poly_with_certificate("lambda").unwrap();
        assert_eq!(cert1.poly.degree(), Some(1));
        assert!(verify_charpoly_certificate(&m1, &cert1).is_ok());

        // 0x0 matrix test
        let m0 = Matrix::zeros(0, 0).unwrap();
        let cert0 = m0.char_poly_with_certificate("lambda").unwrap();
        assert_eq!(cert0.poly.degree(), Some(0));
        assert!(verify_charpoly_certificate(&m0, &cert0).is_ok());

        // Hand-planted repeated-eigenvalue case, independent of the generator:
        // det(lambda I - I_4) = (lambda - 1)^4.
        let identity4 = Matrix::eye(4).unwrap();
        let planted_identity4 = CharpolyCertificate {
            poly: UnivariatePoly::new(
                Symbol::new("lambda"),
                vec![
                    BigRational::one(),
                    BigRational::from_integer((-4).into()),
                    BigRational::from_integer(6.into()),
                    BigRational::from_integer((-4).into()),
                    BigRational::one(),
                ],
            ),
        };
        assert!(verify_charpoly_certificate(&identity4, &planted_identity4).is_ok());

        // The reference lane is not restricted by the Laplace determinant's 8x8 cap.
        let m9 = Matrix::zeros(9, 9).unwrap();
        let cert9 = m9.char_poly_with_certificate("lambda").unwrap();
        assert_eq!(cert9.poly.degree(), Some(9));
        assert!(verify_charpoly_certificate(&m9, &cert9).is_ok());
    }

    #[test]
    fn charpoly_verifier_rejects_mutants() {
        let matrix = Matrix::new(2, 2, vec![num(1), num(2), num(3), num(4)]).unwrap();
        let valid_cert = matrix.char_poly_with_certificate("lambda").unwrap();

        // Mutant 1: Non-monic leading coefficient
        let mut bad_coeffs = valid_cert.poly.coeffs.clone();
        bad_coeffs[2] = BigRational::from_integer(2.into());
        let bad_cert1 = CharpolyCertificate {
            poly: UnivariatePoly::new(Symbol::new("lambda"), bad_coeffs),
        };
        assert!(verify_charpoly_certificate(&matrix, &bad_cert1).is_err());

        // Mutant 2: Wrong degree
        let mut bad_coeffs2 = valid_cert.poly.coeffs.clone();
        bad_coeffs2.push(BigRational::one());
        let bad_cert2 = CharpolyCertificate {
            poly: UnivariatePoly::new(Symbol::new("lambda"), bad_coeffs2),
        };
        assert!(verify_charpoly_certificate(&matrix, &bad_cert2).is_err());

        // Mutant 3: Wrong trace / subleading coefficient
        let mut bad_coeffs3 = valid_cert.poly.coeffs.clone();
        bad_coeffs3[1] = BigRational::zero();
        let bad_cert3 = CharpolyCertificate {
            poly: UnivariatePoly::new(Symbol::new("lambda"), bad_coeffs3),
        };
        assert!(verify_charpoly_certificate(&matrix, &bad_cert3).is_err());

        // Mutant 4: Wrong constant term
        let mut bad_coeffs4 = valid_cert.poly.coeffs.clone();
        bad_coeffs4[0] = BigRational::zero();
        let bad_cert4 = CharpolyCertificate {
            poly: UnivariatePoly::new(Symbol::new("lambda"), bad_coeffs4),
        };
        assert!(verify_charpoly_certificate(&matrix, &bad_cert4).is_err());

        // Mutant 5: Cayley-Hamilton failure (e.g. wrong intermediate coefficient on 3x3)
        let m3 = Matrix::new(
            3,
            3,
            vec![
                num(1),
                num(0),
                num(0),
                num(0),
                num(2),
                num(0),
                num(0),
                num(0),
                num(3),
            ],
        )
        .unwrap();
        let cert3 = m3.char_poly_with_certificate("lambda").unwrap();
        // trace = 6, det = 6. True poly: lambda^3 - 6 lambda^2 + 11 lambda - 6
        // If we tamper the middle coefficient (11 -> 10), trace and det still match,
        // but Cayley-Hamilton fails!
        let mut tampered_coeffs = cert3.poly.coeffs.clone();
        tampered_coeffs[1] = BigRational::from_integer(10.into());
        let bad_cert5 = CharpolyCertificate {
            poly: UnivariatePoly::new(Symbol::new("lambda"), tampered_coeffs),
        };
        assert!(verify_charpoly_certificate(&m3, &bad_cert5).is_err());

        // Cayley-Hamilton, trace, and determinant anchors do not characterize the
        // characteristic polynomial when the minimal polynomial has lower degree.
        // For the zero matrix, lambda^3 + lambda satisfies every old check.
        let zero3 = Matrix::zeros(3, 3).unwrap();
        let old_verifier_bypass = CharpolyCertificate {
            poly: UnivariatePoly::new(
                Symbol::new("lambda"),
                vec![
                    BigRational::zero(),
                    BigRational::one(),
                    BigRational::zero(),
                    BigRational::one(),
                ],
            ),
        };
        assert!(verify_charpoly_certificate(&zero3, &old_verifier_bypass).is_err());
    }

    #[test]
    fn matrix_polynomial_evaluation_preflights_shape_and_aggregate_work() {
        let malformed = UnivariatePoly {
            gen_sym: Symbol::new("lambda"),
            coeffs: Vec::new(),
        };
        let scalar = Matrix::new(1, 1, vec![num(1)]).unwrap();
        assert!(matches!(
            scalar.eval_poly(&malformed),
            Err(MatrixError::InvalidPolynomial(_))
        ));

        let large_matrix = Matrix::zeros(100, 100).unwrap();
        let high_degree =
            UnivariatePoly::monomial(Symbol::new("lambda"), BigRational::one(), 1_000).unwrap();
        assert!(matches!(
            large_matrix.eval_poly(&high_degree),
            Err(MatrixError::ResourceLimit(_))
        ));
    }

    #[test]
    fn test_matrix_diag_and_shape_predicates() {
        let diag_mat = Matrix::diag(vec![num(1), num(2), num(3)]).unwrap();
        assert_eq!(diag_mat.rows(), 3);
        assert_eq!(diag_mat.cols(), 3);
        assert!(diag_mat.is_diagonal());
        assert!(diag_mat.is_symmetric());
        assert!(diag_mat.is_upper_triangular());
        assert!(diag_mat.is_lower_triangular());

        let sym_mat = Matrix::new(2, 2, vec![num(1), num(4), num(4), num(5)]).unwrap();
        assert!(sym_mat.is_symmetric());
        assert!(!sym_mat.is_diagonal());
        assert!(!sym_mat.is_upper_triangular());
        assert!(!sym_mat.is_lower_triangular());

        let upper = Matrix::new(2, 2, vec![num(1), num(2), num(0), num(3)]).unwrap();
        assert!(upper.is_upper_triangular());
        assert!(!upper.is_lower_triangular());
        assert!(!upper.is_diagonal());
        assert!(!upper.is_symmetric());

        let lower = Matrix::new(2, 2, vec![num(1), num(0), num(2), num(3)]).unwrap();
        assert!(lower.is_lower_triangular());
        assert!(!lower.is_upper_triangular());
        assert!(!lower.is_diagonal());
        assert!(!lower.is_symmetric());
    }
}
