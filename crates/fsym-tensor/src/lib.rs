//! # fsym-tensor
//!
//! Symbolic tensors, abstract index notation, tensor contractions, and differential geometry.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("Index mismatch in tensor contraction")]
    IndexMismatch,
    #[error("Tensor rank mismatch: expected {0}, got {1}")]
    RankMismatch(usize, usize),
}

/// Tensor index variance: Contravariant (upper index) or Covariant (lower index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexVariance {
    Upper,
    Lower,
}

/// Symbolic tensor index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorIndex {
    pub symbol: Symbol,
    pub variance: IndexVariance,
}

impl TensorIndex {
    pub fn upper(name: impl Into<String>) -> Self {
        Self {
            symbol: Symbol::new(name),
            variance: IndexVariance::Upper,
        }
    }

    pub fn lower(name: impl Into<String>) -> Self {
        Self {
            symbol: Symbol::new(name),
            variance: IndexVariance::Lower,
        }
    }
}

/// Symbolic tensor expression with named indices: T^{\mu}_{\nu}.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorExpr {
    pub name: String,
    pub indices: Vec<TensorIndex>,
    pub components: Option<Vec<Expr>>,
}

impl TensorExpr {
    pub fn new(name: impl Into<String>, indices: Vec<TensorIndex>) -> Self {
        Self {
            name: name.into(),
            indices,
            components: None,
        }
    }

    pub fn rank(&self) -> usize {
        self.indices.len()
    }
}

impl fmt::Display for TensorExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        for idx in &self.indices {
            match idx.variance {
                IndexVariance::Upper => write!(f, "^{}", idx.symbol)?,
                IndexVariance::Lower => write!(f, "_{}", idx.symbol)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_indices() {
        let t = TensorExpr::new(
            "T",
            vec![TensorIndex::upper("mu"), TensorIndex::lower("nu")],
        );
        assert_eq!(t.rank(), 2);
        assert_eq!(format!("{}", t), "T^mu_nu");
    }
}
