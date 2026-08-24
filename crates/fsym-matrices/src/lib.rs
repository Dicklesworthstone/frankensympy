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
const MAX_DETERMINANT_DIMENSION: usize = 8;
const MAX_CHARACTERISTIC_POLYNOMIAL_DIMENSION: usize = 32;
const MAX_MATRIX_MULTIPLICATION_OPS: u128 = 10_000_000;
const MAX_RREF_OPS: u128 = 10_000_000;

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
    #[error("Matrix operation exceeds a supported resource bound: {0}")]
    ResourceLimit(String),
    #[error("A symbolic pivot or determinant may be zero; an unconditional result is unsafe")]
    SymbolicZeroUndetermined,
    #[error("Exact matrix division by zero")]
    DivisionByZero,
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

        let mut basis = Vec::new();
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
    fn zero_dimension_metered_operations_do_not_issue_zero_charges() {
        let empty = Matrix::new(0, 0, Vec::new()).unwrap();
        let mut budget = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(1, 0));
        assert_eq!(empty.metered_matmul(&empty, &mut budget).unwrap(), empty);
        let (reduced, pivots) = empty.metered_rref(&mut budget).unwrap();
        assert_eq!(reduced, empty);
        assert!(pivots.is_empty());
    }
}
