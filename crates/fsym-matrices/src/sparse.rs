//! Sparse matrix representation over $\mathbb{Q}$ and symbolic expressions (WS10).

#![forbid(unsafe_code)]

use crate::{Matrix, MatrixError};
use fsym_core::Expr;
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;

const SPARSE_MATRIX_SCHEMA_VERSION: u32 = 1;
const MAX_SPARSE_MATRIX_ENTRIES: usize = 262_144;
const MAX_SPARSE_TERM_PRODUCTS: u64 = 1_000_000;

/// Exact sparse matrix stored as a sorted map of non-zero entries: `(row, col) -> Expr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), Expr>,
}

#[derive(Serialize)]
struct SparseMatrixWireRef<'a> {
    schema_version: u32,
    rows: usize,
    cols: usize,
    entries: Vec<SparseEntryWireRef<'a>>,
}

#[derive(Serialize)]
struct SparseEntryWireRef<'a> {
    row: usize,
    col: usize,
    value: &'a Expr,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SparseMatrixWire {
    schema_version: u32,
    rows: usize,
    cols: usize,
    entries: BoundedSparseEntries,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SparseEntryWire {
    row: usize,
    col: usize,
    value: Expr,
}

struct BoundedSparseEntries(Vec<SparseEntryWire>);

impl<'de> Deserialize<'de> for BoundedSparseEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BoundedSparseEntriesVisitor;

        impl<'de> Visitor<'de> for BoundedSparseEntriesVisitor {
            type Value = BoundedSparseEntries;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "a sparse-entry sequence with at most {MAX_SPARSE_MATRIX_ENTRIES} entries"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let hinted = sequence.size_hint().unwrap_or(0);
                if hinted > MAX_SPARSE_MATRIX_ENTRIES {
                    return Err(serde::de::Error::custom(format!(
                        "sparse matrix exceeds the entry limit of {MAX_SPARSE_MATRIX_ENTRIES}"
                    )));
                }
                let mut entries = Vec::with_capacity(hinted.min(MAX_SPARSE_MATRIX_ENTRIES));
                loop {
                    if entries.len() == MAX_SPARSE_MATRIX_ENTRIES {
                        if sequence.next_element::<IgnoredAny>()?.is_some() {
                            return Err(serde::de::Error::custom(format!(
                                "sparse matrix exceeds the entry limit of {MAX_SPARSE_MATRIX_ENTRIES}"
                            )));
                        }
                        break;
                    }
                    let Some(entry) = sequence.next_element()? else {
                        break;
                    };
                    entries.push(entry);
                }
                Ok(BoundedSparseEntries(entries))
            }
        }

        deserializer.deserialize_seq(BoundedSparseEntriesVisitor)
    }
}

impl Serialize for SparseMatrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate_shape().map_err(serde::ser::Error::custom)?;
        let entries = self
            .entries
            .iter()
            .map(|(&(row, col), value)| SparseEntryWireRef { row, col, value })
            .collect();
        SparseMatrixWireRef {
            schema_version: SPARSE_MATRIX_SCHEMA_VERSION,
            rows: self.rows,
            cols: self.cols,
            entries,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SparseMatrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SparseMatrixWire::deserialize(deserializer)?;
        if wire.schema_version != SPARSE_MATRIX_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported sparse-matrix schema version {}",
                wire.schema_version
            )));
        }
        let mut entries = BTreeMap::new();
        for entry in wire.entries.0 {
            if entry.row >= wire.rows || entry.col >= wire.cols {
                return Err(serde::de::Error::custom(format!(
                    "sparse entry ({}, {}) lies outside shape {}x{}",
                    entry.row, entry.col, wire.rows, wire.cols
                )));
            }
            if entry.value.is_zero() {
                return Err(serde::de::Error::custom(
                    "sparse canonical wire cannot contain explicit zero entries",
                ));
            }
            if entries
                .insert((entry.row, entry.col), entry.value)
                .is_some()
            {
                return Err(serde::de::Error::custom(
                    "sparse canonical wire contains a duplicate coordinate",
                ));
            }
        }
        Self::new(wire.rows, wire.cols, entries).map_err(serde::de::Error::custom)
    }
}

impl SparseMatrix {
    fn validate_shape(&self) -> Result<(), MatrixError> {
        if self.entries.len() > MAX_SPARSE_MATRIX_ENTRIES {
            return Err(MatrixError::EntryLimitExceeded(self.entries.len()));
        }
        for (&(row, col), value) in &self.entries {
            if row >= self.rows || col >= self.cols {
                return Err(MatrixError::OutOfBounds(row, col, self.rows, self.cols));
            }
            if value.is_zero() {
                return Err(MatrixError::ResourceLimit(
                    "sparse canonical form contains an explicit zero".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Canonical non-zero entries.
    pub fn entries(&self) -> &BTreeMap<(usize, usize), Expr> {
        &self.entries
    }

    /// Create a new sparse matrix from non-zero entries.
    pub fn new(
        rows: usize,
        cols: usize,
        raw_entries: BTreeMap<(usize, usize), Expr>,
    ) -> Result<Self, MatrixError> {
        if raw_entries.len() > MAX_SPARSE_MATRIX_ENTRIES {
            return Err(MatrixError::EntryLimitExceeded(raw_entries.len()));
        }
        let mut entries = BTreeMap::new();
        for ((r, c), val) in raw_entries {
            if r >= rows || c >= cols {
                return Err(MatrixError::OutOfBounds(r, c, rows, cols));
            }
            if !val.is_zero() {
                entries.insert((r, c), val);
            }
        }
        Ok(Self {
            rows,
            cols,
            entries,
        })
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
    pub fn eye(n: usize) -> Result<Self, MatrixError> {
        if n > MAX_SPARSE_MATRIX_ENTRIES {
            return Err(MatrixError::EntryLimitExceeded(n));
        }
        let mut entries = BTreeMap::new();
        for i in 0..n {
            entries.insert((i, i), Expr::from_i64(1));
        }
        Ok(Self {
            rows: n,
            cols: n,
            entries,
        })
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
    pub fn to_dense(&self) -> Result<Matrix, MatrixError> {
        self.validate_shape()?;
        let mut dense = Matrix::zeros(self.rows, self.cols)?;
        for (&(r, c), val) in &self.entries {
            dense.data[r * self.cols + c] = val.clone();
        }
        Ok(dense)
    }

    /// Matrix addition for sparse matrices.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        self.validate_shape()?;
        other.validate_shape()?;
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
        Self::new(self.rows, self.cols, new_entries)
    }

    /// Matrix multiplication for sparse matrices.
    pub fn matmul(&self, other: &Self) -> Result<Self, MatrixError> {
        self.validate_shape()?;
        other.validate_shape()?;
        if self.cols != other.rows {
            return Err(MatrixError::ShapeMismatch(
                self.rows, self.cols, other.rows, other.cols,
            ));
        }
        let mut result_entries = BTreeMap::new();
        let mut term_products = 0u64;
        for (&(r, k1), v1) in &self.entries {
            for (&(_, c), v2) in other.entries.range((k1, 0)..=(k1, usize::MAX)) {
                term_products = term_products.checked_add(1).ok_or_else(|| {
                    MatrixError::ResourceLimit(
                        "sparse multiplication term-product count overflowed".to_string(),
                    )
                })?;
                if term_products > MAX_SPARSE_TERM_PRODUCTS {
                    return Err(MatrixError::ResourceLimit(format!(
                        "sparse multiplication exceeds the term-product limit of {MAX_SPARSE_TERM_PRODUCTS}"
                    )));
                }
                let prod = Matrix::exact_mul(v1, v2);
                let is_new_coordinate = !result_entries.contains_key(&(r, c));
                if is_new_coordinate && result_entries.len() == MAX_SPARSE_MATRIX_ENTRIES {
                    return Err(MatrixError::EntryLimitExceeded(result_entries.len() + 1));
                }
                let acc = result_entries
                    .entry((r, c))
                    .or_insert_with(|| Expr::from_i64(0));
                *acc = Matrix::exact_add(acc, &prod);
                if acc.is_zero() {
                    result_entries.remove(&(r, c));
                }
            }
        }
        Self::new(self.rows, other.cols, result_entries)
    }
}
