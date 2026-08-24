//! # fsym-tensor
//!
//! Symbolic tensors, abstract index notation, tensor contractions, and differential geometry (WS20).

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("Index mismatch in tensor contraction: {0}")]
    IndexMismatch(String),
    #[error("Tensor rank mismatch: expected {0}, got {1}")]
    RankMismatch(usize, usize),
}

/// Tensor index variance: Contravariant (upper index) or Covariant (lower index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexVariance {
    Upper,
    Lower,
}

impl IndexVariance {
    pub fn flip(&self) -> Self {
        match self {
            IndexVariance::Upper => IndexVariance::Lower,
            IndexVariance::Lower => IndexVariance::Upper,
        }
    }
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

    pub fn flip_variance(&self) -> Self {
        Self {
            symbol: self.symbol.clone(),
            variance: self.variance.flip(),
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

    /// Raises or lowers an index using metric tensor contraction: g_{\mu \alpha} T^{\alpha}_{\nu} = T_{\mu \nu}.
    pub fn contract_index(
        &self,
        target_sym: &Symbol,
        new_name: impl Into<String>,
    ) -> Result<Self, TensorError> {
        let pos = self
            .indices
            .iter()
            .position(|idx| &idx.symbol == target_sym)
            .ok_or_else(|| {
                TensorError::IndexMismatch(format!("Index {} not found in tensor", target_sym))
            })?;

        let mut new_indices = self.indices.clone();
        new_indices[pos] = new_indices[pos].flip_variance();

        Ok(Self {
            name: new_name.into(),
            indices: new_indices,
            components: None,
        })
    }

    /// Outer product with another tensor: (T \otimes S)^{\mu}_{\nu \alpha \beta}.
    pub fn outer_product(&self, other: &Self, new_name: impl Into<String>) -> Self {
        let mut new_indices = self.indices.clone();
        new_indices.extend(other.indices.clone());
        Self {
            name: new_name.into(),
            indices: new_indices,
            components: None,
        }
    }

    /// Contraction of an upper index in `self` with a matching lower index in `self` (Einstein trace).
    pub fn self_contract(
        &self,
        upper_sym: &Symbol,
        lower_sym: &Symbol,
        new_name: impl Into<String>,
    ) -> Result<Self, TensorError> {
        let up_pos = self
            .indices
            .iter()
            .position(|idx| &idx.symbol == upper_sym && idx.variance == IndexVariance::Upper)
            .ok_or_else(|| {
                TensorError::IndexMismatch(format!("Upper index {} not found", upper_sym))
            })?;

        let low_pos = self
            .indices
            .iter()
            .position(|idx| &idx.symbol == lower_sym && idx.variance == IndexVariance::Lower)
            .ok_or_else(|| {
                TensorError::IndexMismatch(format!("Lower index {} not found", lower_sym))
            })?;

        let mut new_indices = Vec::new();
        for (i, idx) in self.indices.iter().enumerate() {
            if i != up_pos && i != low_pos {
                new_indices.push(idx.clone());
            }
        }

        Ok(Self {
            name: new_name.into(),
            indices: new_indices,
            components: None,
        })
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
    fn test_tensor_indices_and_contraction() {
        let t = TensorExpr::new(
            "T",
            vec![TensorIndex::upper("mu"), TensorIndex::lower("nu")],
        );
        assert_eq!(t.rank(), 2);
        assert_eq!(format!("{}", t), "T^mu_nu");

        // Contract/lower index mu
        let lowered = t.contract_index(&Symbol::new("mu"), "T").unwrap();
        assert_eq!(format!("{}", lowered), "T_mu_nu");

        // Outer product and self contraction
        let s = TensorExpr::new(
            "S",
            vec![TensorIndex::upper("alpha"), TensorIndex::lower("beta")],
        );
        let prod = t.outer_product(&s, "P");
        assert_eq!(prod.rank(), 4);
        assert_eq!(format!("{}", prod), "P^mu_nu^alpha_beta");

        // Self contraction of mu and nu in P
        let trace = prod
            .self_contract(&Symbol::new("mu"), &Symbol::new("nu"), "Trace")
            .unwrap();
        assert_eq!(trace.rank(), 2);
        assert_eq!(format!("{}", trace), "Trace^alpha_beta");
    }
}
