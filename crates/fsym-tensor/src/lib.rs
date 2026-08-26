//! # fsym-tensor
//!
//! Symbolic tensors, abstract index notation, tensor contractions, and differential geometry (WS20).

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use fsym_simplify::simplify;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TensorError {
    #[error("Index mismatch in tensor contraction: {0}")]
    IndexMismatch(String),
    #[error("Tensor rank mismatch: expected {0}, got {1}")]
    RankMismatch(usize, usize),
    #[error("Tensor dimension mismatch: {0} vs {1}")]
    DimensionMismatch(usize, usize),
    #[error(
        "Tensor component count mismatch: expected {expected} for dimension {dimension} and rank {rank}, got {actual}"
    )]
    ComponentCountMismatch {
        expected: usize,
        actual: usize,
        dimension: usize,
        rank: usize,
    },
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

fn decode_multi_index(flat_idx: usize, rank: usize, dim: usize) -> Vec<usize> {
    if rank == 0 {
        return Vec::new();
    }
    let mut indices = vec![0; rank];
    let mut curr = flat_idx;
    for i in (0..rank).rev() {
        indices[i] = curr % dim;
        curr /= dim;
    }
    indices
}

fn encode_multi_index(indices: &[usize], dim: usize) -> usize {
    let mut flat = 0;
    for &idx in indices {
        flat = flat * dim + idx;
    }
    flat
}

/// Symbolic tensor expression with named indices and optional concrete components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorExpr {
    pub name: String,
    pub dimension: usize,
    pub indices: Vec<TensorIndex>,
    pub components: Option<Vec<Expr>>,
}

impl TensorExpr {
    pub fn new(name: impl Into<String>, indices: Vec<TensorIndex>) -> Self {
        Self {
            name: name.into(),
            dimension: 4,
            indices,
            components: None,
        }
    }

    pub fn with_components(
        name: impl Into<String>,
        dimension: usize,
        indices: Vec<TensorIndex>,
        components: Vec<Expr>,
    ) -> Result<Self, TensorError> {
        let rank = indices.len();
        let expected = if rank == 0 {
            1
        } else {
            dimension.checked_pow(rank as u32).ok_or_else(|| {
                TensorError::ComponentCountMismatch {
                    expected: usize::MAX,
                    actual: components.len(),
                    dimension,
                    rank,
                }
            })?
        };
        if components.len() != expected {
            return Err(TensorError::ComponentCountMismatch {
                expected,
                actual: components.len(),
                dimension,
                rank,
            });
        }
        Ok(Self {
            name: name.into(),
            dimension,
            indices,
            components: Some(components),
        })
    }

    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Raises or lowers an index variance symbolically.
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
            dimension: self.dimension,
            indices: new_indices,
            components: self.components.clone(),
        })
    }

    /// Outer product with another tensor: (T \otimes S)^{\mu}_{\nu \alpha \beta}.
    pub fn outer_product(
        &self,
        other: &Self,
        new_name: impl Into<String>,
    ) -> Result<Self, TensorError> {
        if self.dimension != other.dimension
            && self.components.is_some()
            && other.components.is_some()
        {
            return Err(TensorError::DimensionMismatch(
                self.dimension,
                other.dimension,
            ));
        }
        let mut new_indices = self.indices.clone();
        new_indices.extend(other.indices.clone());

        let new_components = match (&self.components, &other.components) {
            (Some(c1), Some(c2)) => {
                let mut comp = Vec::with_capacity(c1.len() * c2.len());
                for a in c1 {
                    for b in c2 {
                        comp.push(simplify(&(a.clone() * b.clone())));
                    }
                }
                Some(comp)
            }
            _ => None,
        };

        Ok(Self {
            name: new_name.into(),
            dimension: self.dimension,
            indices: new_indices,
            components: new_components,
        })
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

        if up_pos == low_pos {
            return Err(TensorError::IndexMismatch(format!(
                "Cannot contract an index {} with itself at the same position",
                upper_sym
            )));
        }

        let mut new_indices = Vec::new();
        for (i, idx) in self.indices.iter().enumerate() {
            if i != up_pos && i != low_pos {
                new_indices.push(idx.clone());
            }
        }

        let out_rank = new_indices.len();
        let new_components = if let Some(components) = &self.components {
            let dim = self.dimension;
            let out_len = if out_rank == 0 {
                1
            } else {
                dim.pow(out_rank as u32)
            };
            let in_rank = self.rank();
            let mut out_comp = Vec::with_capacity(out_len);

            for out_flat in 0..out_len {
                let out_multi = decode_multi_index(out_flat, out_rank, dim);
                let mut sum_terms = Vec::with_capacity(dim);

                for d in 0..dim {
                    let mut in_multi = vec![0; in_rank];
                    in_multi[up_pos] = d;
                    in_multi[low_pos] = d;

                    let mut out_idx = 0;
                    for (i, slot) in in_multi.iter_mut().enumerate() {
                        if i != up_pos && i != low_pos {
                            *slot = out_multi[out_idx];
                            out_idx += 1;
                        }
                    }

                    let in_flat = encode_multi_index(&in_multi, dim);
                    sum_terms.push(components[in_flat].clone());
                }

                let sum_expr = Expr::Add(sum_terms);
                out_comp.push(simplify(&sum_expr));
            }
            Some(out_comp)
        } else {
            None
        };

        Ok(Self {
            name: new_name.into(),
            dimension: self.dimension,
            indices: new_indices,
            components: new_components,
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
        let prod = t.outer_product(&s, "P").unwrap();
        assert_eq!(prod.rank(), 4);
        assert_eq!(format!("{}", prod), "P^mu_nu^alpha_beta");

        // Self contraction of mu and nu in P
        let trace = prod
            .self_contract(&Symbol::new("mu"), &Symbol::new("nu"), "Trace")
            .unwrap();
        assert_eq!(trace.rank(), 2);
        assert_eq!(format!("{}", trace), "Trace^alpha_beta");
    }

    #[test]
    fn test_tensor_component_einstein_trace_and_matrix_multiplication() {
        let dim = 4;
        // Kronecker delta / Identity tensor: delta^\mu_\nu
        let mut delta_components = Vec::with_capacity(16);
        for i in 0..dim {
            for j in 0..dim {
                delta_components.push(if i == j {
                    Expr::from_i64(1)
                } else {
                    Expr::from_i64(0)
                });
            }
        }
        let delta = TensorExpr::with_components(
            "delta",
            dim,
            vec![TensorIndex::upper("mu"), TensorIndex::lower("nu")],
            delta_components,
        )
        .unwrap();

        // Einstein trace delta^\mu_\mu = 1 + 1 + 1 + 1 = 4
        let trace_delta = delta
            .self_contract(&Symbol::new("mu"), &Symbol::new("nu"), "Trace")
            .unwrap();
        assert_eq!(trace_delta.rank(), 0);
        let comp = trace_delta.components.unwrap();
        assert_eq!(comp.len(), 1);
        assert_eq!(comp[0], Expr::from_i64(4));

        // 4D 4-vectors: A^\mu = (1, 2, 3, 4), B_\nu = (2, 0, -1, 3)
        let a_vec = TensorExpr::with_components(
            "A",
            dim,
            vec![TensorIndex::upper("mu")],
            vec![
                Expr::from_i64(1),
                Expr::from_i64(2),
                Expr::from_i64(3),
                Expr::from_i64(4),
            ],
        )
        .unwrap();
        let b_vec = TensorExpr::with_components(
            "B",
            dim,
            vec![TensorIndex::lower("nu")],
            vec![
                Expr::from_i64(2),
                Expr::from_i64(0),
                Expr::from_i64(-1),
                Expr::from_i64(3),
            ],
        )
        .unwrap();

        let outer = a_vec.outer_product(&b_vec, "P").unwrap();
        assert_eq!(outer.rank(), 2);
        let dot = outer
            .self_contract(&Symbol::new("mu"), &Symbol::new("nu"), "Dot")
            .unwrap();
        // Dot product: 1*2 + 2*0 + 3*(-1) + 4*3 = 2 + 0 - 3 + 12 = 11
        assert_eq!(dot.rank(), 0);
        assert_eq!(dot.components.unwrap()[0], Expr::from_i64(11));
    }

    #[test]
    fn test_tensor_dimension_and_count_validation() {
        let invalid_count = TensorExpr::with_components(
            "T",
            3,
            vec![TensorIndex::upper("mu"), TensorIndex::lower("nu")],
            vec![Expr::from_i64(1); 4], // expected 3^2 = 9
        );
        assert!(matches!(
            invalid_count,
            Err(TensorError::ComponentCountMismatch {
                expected: 9,
                actual: 4,
                ..
            })
        ));

        let t3 = TensorExpr::with_components(
            "T3",
            3,
            vec![TensorIndex::upper("mu")],
            vec![Expr::from_i64(1); 3],
        )
        .unwrap();
        let t4 = TensorExpr::with_components(
            "T4",
            4,
            vec![TensorIndex::lower("nu")],
            vec![Expr::from_i64(1); 4],
        )
        .unwrap();

        assert_eq!(
            t3.outer_product(&t4, "P"),
            Err(TensorError::DimensionMismatch(3, 4))
        );
    }
}
