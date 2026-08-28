//! # fsym-tensor
//!
//! Symbolic tensors, abstract index notation, tensor contractions, and differential geometry (WS20).

#![forbid(unsafe_code)]

use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_simplify::{SimplifyError, try_simplify};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use thiserror::Error;

const MAX_TENSOR_COMPONENTS: usize = 262_144;
const MAX_COMPONENT_EXPRESSION_NODES: usize = 262_144;
const MAX_COMPONENT_EXPRESSION_DEPTH: usize = 64;
const MAX_COMPONENT_EXPRESSION_FANOUT: usize = 4_096;

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
    #[error(
        "Tensor component count for dimension {dimension} and rank {rank} exceeds the supported range"
    )]
    ComponentCountOverflow { dimension: usize, rank: usize },
    #[error("Tensor has {0} components, exceeding the limit of {MAX_TENSOR_COMPONENTS}")]
    ComponentLimitExceeded(usize),
    #[error("Raising or lowering concrete tensor components requires an explicit metric")]
    ConcreteVarianceChangeRequiresMetric,
    #[error(
        "Tensor component expression depth {actual} exceeds the limit of {MAX_COMPONENT_EXPRESSION_DEPTH}"
    )]
    ComponentExpressionDepthLimitExceeded { actual: usize },
    #[error(
        "Tensor component expressions exceed the aggregate node limit of {MAX_COMPONENT_EXPRESSION_NODES}"
    )]
    ComponentExpressionNodeLimitExceeded,
    #[error(
        "Tensor component expression fanout {actual} exceeds the limit of {MAX_COMPONENT_EXPRESSION_FANOUT}"
    )]
    ComponentExpressionFanoutLimitExceeded { actual: usize },
    #[error("Tensor operation could not reserve storage for {requested} work items")]
    AllocationFailure { requested: usize },
    #[error("Diagonal metric entry at index {index} is zero and cannot be inverted")]
    ZeroDiagonalMetricEntry { index: usize },
    #[error(transparent)]
    Simplification(#[from] SimplifyError),
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

fn try_reserve<T>(items: &mut Vec<T>, additional: usize) -> Result<(), TensorError> {
    let requested = items
        .len()
        .checked_add(additional)
        .ok_or(TensorError::AllocationFailure {
            requested: usize::MAX,
        })?;
    items
        .try_reserve_exact(additional)
        .map_err(|_| TensorError::AllocationFailure { requested })
}

fn decode_multi_index(flat_idx: usize, rank: usize, dim: usize) -> Result<Vec<usize>, TensorError> {
    if rank == 0 {
        return Ok(Vec::new());
    }
    let mut indices = Vec::new();
    try_reserve(&mut indices, rank)?;
    indices.resize(rank, 0);
    let mut curr = flat_idx;
    for i in (0..rank).rev() {
        indices[i] = curr % dim;
        curr /= dim;
    }
    Ok(indices)
}

fn encode_multi_index(indices: &[usize], dim: usize) -> usize {
    let mut flat = 0;
    for &idx in indices {
        flat = flat * dim + idx;
    }
    flat
}

/// Symbolic tensor expression with named indices and optional concrete components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorExpr {
    pub name: String,
    pub dimension: usize,
    pub indices: Vec<TensorIndex>,
    pub components: Option<Vec<Expr>>,
}

#[derive(Serialize)]
struct TensorExprWireRef<'a> {
    name: &'a str,
    dimension: usize,
    indices: &'a [TensorIndex],
    components: Option<&'a [Expr]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorExprWire {
    name: String,
    dimension: usize,
    indices: Vec<TensorIndex>,
    components: Option<Vec<Expr>>,
}

impl Serialize for TensorExpr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate_components()
            .map_err(serde::ser::Error::custom)?;
        TensorExprWireRef {
            name: &self.name,
            dimension: self.dimension,
            indices: &self.indices,
            components: self.components.as_deref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TensorExpr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TensorExprWire::deserialize(deserializer)?;
        let tensor = Self {
            name: wire.name,
            dimension: wire.dimension,
            indices: wire.indices,
            components: wire.components,
        };
        tensor
            .validate_components()
            .map_err(serde::de::Error::custom)?;
        Ok(tensor)
    }
}

impl TensorExpr {
    fn expected_component_count(dimension: usize, rank: usize) -> Result<usize, TensorError> {
        let count = if rank == 0 {
            1
        } else {
            let exponent = u32::try_from(rank)
                .map_err(|_| TensorError::ComponentCountOverflow { dimension, rank })?;
            dimension
                .checked_pow(exponent)
                .ok_or(TensorError::ComponentCountOverflow { dimension, rank })?
        };
        if count > MAX_TENSOR_COMPONENTS {
            return Err(TensorError::ComponentLimitExceeded(count));
        }
        Ok(count)
    }

    fn validate_components(&self) -> Result<(), TensorError> {
        let Some(components) = &self.components else {
            return Ok(());
        };
        let rank = self.rank();
        let expected = Self::expected_component_count(self.dimension, rank)?;
        if components.len() != expected {
            return Err(TensorError::ComponentCountMismatch {
                expected,
                actual: components.len(),
                dimension: self.dimension,
                rank,
            });
        }
        Self::validate_component_expressions(components)?;
        Ok(())
    }

    fn validate_component_expressions(components: &[Expr]) -> Result<(), TensorError> {
        let mut visited = 0usize;

        for root in components {
            Self::validate_component_expression(root, &mut visited)?;
        }

        Ok(())
    }

    fn validate_component_expression(root: &Expr, visited: &mut usize) -> Result<(), TensorError> {
        Self::validate_component_expression_at(root, 0, visited)
    }

    fn validate_component_expression_at(
        expr: &Expr,
        depth: usize,
        visited: &mut usize,
    ) -> Result<(), TensorError> {
        if depth > MAX_COMPONENT_EXPRESSION_DEPTH {
            return Err(TensorError::ComponentExpressionDepthLimitExceeded { actual: depth });
        }
        *visited = visited
            .checked_add(1)
            .ok_or(TensorError::ComponentExpressionNodeLimitExceeded)?;
        if *visited > MAX_COMPONENT_EXPRESSION_NODES {
            return Err(TensorError::ComponentExpressionNodeLimitExceeded);
        }

        let children: &[Expr] = match expr {
            Expr::Add(items) | Expr::Mul(items) | Expr::Function(_, items) => items,
            Expr::Pow(base, exponent) => {
                Self::validate_component_expression_at(base, depth + 1, visited)?;
                Self::validate_component_expression_at(exponent, depth + 1, visited)?;
                return Ok(());
            }
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => return Ok(()),
        };

        if children.len() > MAX_COMPONENT_EXPRESSION_FANOUT {
            return Err(TensorError::ComponentExpressionFanoutLimitExceeded {
                actual: children.len(),
            });
        }
        for child in children {
            Self::validate_component_expression_at(child, depth + 1, visited)?;
        }

        Ok(())
    }

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
        let expected = Self::expected_component_count(dimension, rank)?;
        if components.len() != expected {
            return Err(TensorError::ComponentCountMismatch {
                expected,
                actual: components.len(),
                dimension,
                rank,
            });
        }
        let tensor = Self {
            name: name.into(),
            dimension,
            indices,
            components: Some(components),
        };
        tensor.validate_components()?;
        Ok(tensor)
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
        self.validate_components()?;
        let pos = self
            .indices
            .iter()
            .position(|idx| &idx.symbol == target_sym)
            .ok_or_else(|| {
                TensorError::IndexMismatch(format!("Index {} not found in tensor", target_sym))
            })?;

        if self.components.is_some() {
            return Err(TensorError::ConcreteVarianceChangeRequiresMetric);
        }

        let mut new_indices = Vec::new();
        try_reserve(&mut new_indices, self.indices.len())?;
        new_indices.extend(self.indices.iter().cloned());
        new_indices[pos] = new_indices[pos].flip_variance();

        Ok(Self {
            name: new_name.into(),
            dimension: self.dimension,
            indices: new_indices,
            components: None,
        })
    }

    /// Outer product with another tensor: (T \otimes S)^{\mu}_{\nu \alpha \beta}.
    pub fn outer_product(
        &self,
        other: &Self,
        new_name: impl Into<String>,
    ) -> Result<Self, TensorError> {
        self.validate_components()?;
        other.validate_components()?;
        if self.dimension != other.dimension {
            return Err(TensorError::DimensionMismatch(
                self.dimension,
                other.dimension,
            ));
        }

        let output_rank =
            self.rank()
                .checked_add(other.rank())
                .ok_or(TensorError::ComponentCountOverflow {
                    dimension: self.dimension,
                    rank: usize::MAX,
                })?;

        let new_components = match (&self.components, &other.components) {
            (Some(c1), Some(c2)) => {
                let output_len = Self::expected_component_count(self.dimension, output_rank)?;
                let mut comp = Vec::new();
                try_reserve(&mut comp, output_len)?;
                let mut output_nodes = 0usize;
                for a in c1 {
                    for b in c2 {
                        let result = try_simplify(&(a.clone() * b.clone()))?;
                        Self::validate_component_expression(&result, &mut output_nodes)?;
                        comp.push(result);
                    }
                }
                Some(comp)
            }
            _ => None,
        };

        let mut new_indices = Vec::new();
        try_reserve(&mut new_indices, output_rank)?;
        new_indices.extend(self.indices.iter().cloned());
        new_indices.extend(other.indices.iter().cloned());

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
        self.validate_components()?;
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
        try_reserve(&mut new_indices, self.rank().saturating_sub(2))?;
        for (i, idx) in self.indices.iter().enumerate() {
            if i != up_pos && i != low_pos {
                new_indices.push(idx.clone());
            }
        }

        let out_rank = new_indices.len();
        let new_components = if let Some(components) = &self.components {
            let dim = self.dimension;
            let out_len = Self::expected_component_count(dim, out_rank)?;
            let in_rank = self.rank();
            let mut out_comp = Vec::new();
            try_reserve(&mut out_comp, out_len)?;
            let mut output_nodes = 0usize;

            for out_flat in 0..out_len {
                let out_multi = decode_multi_index(out_flat, out_rank, dim)?;
                let mut sum_terms = Vec::new();
                try_reserve(&mut sum_terms, dim)?;
                let mut in_multi = Vec::new();
                try_reserve(&mut in_multi, in_rank)?;
                in_multi.resize(in_rank, 0);

                let mut out_idx = 0;
                for (i, slot) in in_multi.iter_mut().enumerate() {
                    if i != up_pos && i != low_pos {
                        *slot = out_multi[out_idx];
                        out_idx += 1;
                    }
                }

                for d in 0..dim {
                    in_multi[up_pos] = d;
                    in_multi[low_pos] = d;

                    let in_flat = encode_multi_index(&in_multi, dim);
                    sum_terms.push(components[in_flat].clone());
                }

                let sum_expr = Expr::Add(sum_terms);
                let result = try_simplify(&sum_expr)?;
                Self::validate_component_expression(&result, &mut output_nodes)?;
                out_comp.push(result);
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

/// Metric tensor $g_{\mu\nu}$ with dimension, covariant components, and inverse metric components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricTensor {
    pub name: String,
    pub dimension: usize,
    pub matrix: Vec<Expr>,
    pub inverse: Vec<Expr>,
}

impl MetricTensor {
    /// Creates a new metric tensor with validated dimension and component lengths.
    pub fn new(
        name: impl Into<String>,
        dimension: usize,
        matrix: Vec<Expr>,
        inverse: Vec<Expr>,
    ) -> Result<Self, TensorError> {
        let expected = dimension
            .checked_mul(dimension)
            .ok_or(TensorError::ComponentCountOverflow { dimension, rank: 2 })?;
        if matrix.len() != expected {
            return Err(TensorError::ComponentCountMismatch {
                expected,
                actual: matrix.len(),
                dimension,
                rank: 2,
            });
        }
        if inverse.len() != expected {
            return Err(TensorError::ComponentCountMismatch {
                expected,
                actual: inverse.len(),
                dimension,
                rank: 2,
            });
        }
        Ok(Self {
            name: name.into(),
            dimension,
            matrix,
            inverse,
        })
    }

    /// Standard 4D Minkowski spacetime metric with signature (-, +, +, +): $\eta = \text{diag}(-1, 1, 1, 1)$.
    pub fn minkowski_4d(name: impl Into<String>) -> Self {
        let diag = vec![
            Expr::from_i64(-1),
            Expr::from_i64(1),
            Expr::from_i64(1),
            Expr::from_i64(1),
        ];
        Self::diagonal(name, diag).expect("valid 4D diagonal")
    }

    /// Euclidean metric of dimension $N$: $g = \text{diag}(1, \dots, 1)$.
    pub fn euclidean(name: impl Into<String>, dimension: usize) -> Result<Self, TensorError> {
        let diag = vec![Expr::from_i64(1); dimension];
        Self::diagonal(name, diag)
    }

    /// Creates a diagonal metric from the given nonzero diagonal elements.
    ///
    /// Exact integer and rational zero entries are refused before matrix storage is allocated or
    /// a reciprocal is constructed. Symbolic entries retain a formal inverse because their
    /// invertibility is not known at this layer.
    pub fn diagonal(name: impl Into<String>, diag_entries: Vec<Expr>) -> Result<Self, TensorError> {
        let dim = diag_entries.len();
        if let Some((index, _)) = diag_entries
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.is_zero())
        {
            return Err(TensorError::ZeroDiagonalMetricEntry { index });
        }
        let expected = dim
            .checked_mul(dim)
            .ok_or(TensorError::ComponentCountOverflow {
                dimension: dim,
                rank: 2,
            })?;
        let mut mat = Vec::new();
        let mut inv = Vec::new();
        try_reserve(&mut mat, expected)?;
        try_reserve(&mut inv, expected)?;

        for (r, entry) in diag_entries.into_iter().enumerate() {
            for c in 0..dim {
                if r == c {
                    mat.push(entry.clone());
                    let inv_entry = match &entry {
                        Expr::Integer(n) => {
                            if n == &BigInt::from(1) {
                                Expr::from_i64(1)
                            } else if n == &BigInt::from(-1) {
                                Expr::from_i64(-1)
                            } else {
                                let r = BigRational::new(1.into(), n.clone());
                                Expr::Rational(r)
                            }
                        }
                        Expr::Rational(r) => {
                            let inv_r = r.recip();
                            if inv_r.is_integer() {
                                Expr::Integer(inv_r.to_integer())
                            } else {
                                Expr::Rational(inv_r)
                            }
                        }
                        _ => try_simplify(&Expr::Pow(
                            std::sync::Arc::new(entry.clone()),
                            std::sync::Arc::new(Expr::from_i64(-1)),
                        ))?,
                    };
                    inv.push(inv_entry);
                } else {
                    mat.push(Expr::from_i64(0));
                    inv.push(Expr::from_i64(0));
                }
            }
        }
        Self::new(name, dim, mat, inv)
    }

    /// Lowers the upper index of a rank-1 contravariant vector: $V_\mu = g_{\mu\nu} V^\nu$.
    pub fn lower_vector(&self, vec_tensor: &TensorExpr) -> Result<TensorExpr, TensorError> {
        if vec_tensor.rank() != 1 {
            return Err(TensorError::RankMismatch(1, vec_tensor.rank()));
        }
        if vec_tensor.indices[0].variance != IndexVariance::Upper {
            return Err(TensorError::IndexMismatch(
                "lower_vector requires a contravariant (upper index) vector".to_string(),
            ));
        }
        if vec_tensor.dimension != self.dimension {
            return Err(TensorError::DimensionMismatch(
                self.dimension,
                vec_tensor.dimension,
            ));
        }
        let Some(comp) = &vec_tensor.components else {
            return vec_tensor.contract_index(
                &vec_tensor.indices[0].symbol,
                format!("{}_low", vec_tensor.name),
            );
        };

        let dim = self.dimension;
        let mut out_comp = Vec::new();
        try_reserve(&mut out_comp, dim)?;
        let mut output_nodes = 0usize;

        for r in 0..dim {
            let mut sum_terms = Vec::new();
            try_reserve(&mut sum_terms, dim)?;
            for (c, v) in comp.iter().enumerate() {
                let g_rc = &self.matrix[r * dim + c];
                if !g_rc.is_zero() && !v.is_zero() {
                    sum_terms.push(Expr::Mul(vec![g_rc.clone(), v.clone()]));
                }
            }
            let sum_expr = if sum_terms.is_empty() {
                Expr::from_i64(0)
            } else {
                Expr::Add(sum_terms)
            };
            let simplified = try_simplify(&sum_expr)?;
            TensorExpr::validate_component_expression(&simplified, &mut output_nodes)?;
            out_comp.push(simplified);
        }

        let new_indices = vec![vec_tensor.indices[0].flip_variance()];
        TensorExpr::with_components(
            format!("{}_low", vec_tensor.name),
            dim,
            new_indices,
            out_comp,
        )
    }

    /// Raises the lower index of a rank-1 covariant covector: $W^\mu = g^{\mu\nu} W_\nu$.
    pub fn raise_covector(&self, covec_tensor: &TensorExpr) -> Result<TensorExpr, TensorError> {
        if covec_tensor.rank() != 1 {
            return Err(TensorError::RankMismatch(1, covec_tensor.rank()));
        }
        if covec_tensor.indices[0].variance != IndexVariance::Lower {
            return Err(TensorError::IndexMismatch(
                "raise_covector requires a covariant (lower index) covector".to_string(),
            ));
        }
        if covec_tensor.dimension != self.dimension {
            return Err(TensorError::DimensionMismatch(
                self.dimension,
                covec_tensor.dimension,
            ));
        }
        let Some(comp) = &covec_tensor.components else {
            return covec_tensor.contract_index(
                &covec_tensor.indices[0].symbol,
                format!("{}_up", covec_tensor.name),
            );
        };

        let dim = self.dimension;
        let mut out_comp = Vec::new();
        try_reserve(&mut out_comp, dim)?;
        let mut output_nodes = 0usize;

        for r in 0..dim {
            let mut sum_terms = Vec::new();
            try_reserve(&mut sum_terms, dim)?;
            for (c, w) in comp.iter().enumerate() {
                let g_inv_rc = &self.inverse[r * dim + c];
                if !g_inv_rc.is_zero() && !w.is_zero() {
                    sum_terms.push(Expr::Mul(vec![g_inv_rc.clone(), w.clone()]));
                }
            }
            let sum_expr = if sum_terms.is_empty() {
                Expr::from_i64(0)
            } else {
                Expr::Add(sum_terms)
            };
            let simplified = try_simplify(&sum_expr)?;
            TensorExpr::validate_component_expression(&simplified, &mut output_nodes)?;
            out_comp.push(simplified);
        }

        let new_indices = vec![covec_tensor.indices[0].flip_variance()];
        TensorExpr::with_components(
            format!("{}_up", covec_tensor.name),
            dim,
            new_indices,
            out_comp,
        )
    }

    /// Computes the metric inner product of two contravariant vectors: $\langle u, v \rangle = g_{\mu\nu} u^\mu v^\nu$.
    pub fn inner_product(&self, u: &TensorExpr, v: &TensorExpr) -> Result<Expr, TensorError> {
        let u_low = self.lower_vector(u)?;
        let Some(u_comp) = &u_low.components else {
            return Err(TensorError::IndexMismatch(
                "Vectors must have concrete components for inner product".to_string(),
            ));
        };
        let Some(v_comp) = &v.components else {
            return Err(TensorError::IndexMismatch(
                "Vectors must have concrete components for inner product".to_string(),
            ));
        };
        let dim = self.dimension;
        let mut terms = Vec::new();
        try_reserve(&mut terms, dim)?;
        for i in 0..dim {
            if !u_comp[i].is_zero() && !v_comp[i].is_zero() {
                terms.push(Expr::Mul(vec![u_comp[i].clone(), v_comp[i].clone()]));
            }
        }
        let sum_expr = if terms.is_empty() {
            Expr::from_i64(0)
        } else {
            Expr::Add(terms)
        };
        Ok(try_simplify(&sum_expr)?)
    }

    /// Computes the metric norm squared of a contravariant vector: $\|v\|^2 = g_{\mu\nu} v^\mu v^\nu$.
    pub fn norm_squared(&self, v: &TensorExpr) -> Result<Expr, TensorError> {
        self.inner_product(v, v)
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

        let symbolic4 = TensorExpr {
            name: "S4".into(),
            dimension: 4,
            indices: vec![TensorIndex::lower("nu")],
            components: None,
        };
        assert_eq!(
            t3.outer_product(&symbolic4, "P"),
            Err(TensorError::DimensionMismatch(3, 4))
        );
    }

    #[test]
    fn concrete_variance_changes_require_a_metric() {
        let vector = TensorExpr::with_components(
            "v",
            2,
            vec![TensorIndex::upper("i")],
            vec![Expr::from_i64(1), Expr::from_i64(2)],
        )
        .unwrap();

        assert_eq!(
            vector.contract_index(&Symbol::new("i"), "lowered"),
            Err(TensorError::ConcreteVarianceChangeRequiresMetric)
        );
    }

    #[test]
    fn malformed_public_component_storage_is_rejected_before_indexing_or_export() {
        let malformed = TensorExpr {
            name: "bad".into(),
            dimension: 2,
            indices: vec![TensorIndex::upper("i"), TensorIndex::lower("j")],
            components: Some(vec![Expr::from_i64(1)]),
        };

        assert!(matches!(
            malformed.self_contract(&Symbol::new("i"), &Symbol::new("j"), "trace"),
            Err(TensorError::ComponentCountMismatch {
                expected: 4,
                actual: 1,
                ..
            })
        ));
        assert!(serde_json::to_string(&malformed).is_err());

        let valid = TensorExpr::with_components(
            "valid",
            2,
            vec![TensorIndex::upper("i"), TensorIndex::lower("j")],
            vec![Expr::from_i64(1); 4],
        )
        .unwrap();
        let mut wire = serde_json::to_value(valid).unwrap();
        wire["components"].as_array_mut().unwrap().truncate(1);
        let error = serde_json::from_value::<TensorExpr>(wire).unwrap_err();
        assert!(error.to_string().contains("expected 4"));
    }

    #[test]
    fn outer_product_refuses_component_amplification_before_allocating_output() {
        let left = TensorExpr::with_components(
            "left",
            64,
            vec![TensorIndex::upper("i"), TensorIndex::lower("j")],
            vec![Expr::from_i64(1); 4_096],
        )
        .unwrap();
        let right = TensorExpr::with_components(
            "right",
            64,
            vec![TensorIndex::upper("k"), TensorIndex::lower("l")],
            vec![Expr::from_i64(1); 4_096],
        )
        .unwrap();

        assert_eq!(
            left.outer_product(&right, "too_large"),
            Err(TensorError::ComponentLimitExceeded(16_777_216))
        );
    }

    #[test]
    fn component_shape_limits_are_checked_before_recursive_work() {
        let mut deep = Expr::from_i64(1);
        for _ in 0..=(MAX_COMPONENT_EXPRESSION_DEPTH + 1) {
            deep = Expr::Function("sin".into(), vec![deep]);
        }
        assert!(matches!(
            TensorExpr::with_components(
                "deep",
                1,
                vec![TensorIndex::upper("i")],
                vec![deep.clone()],
            ),
            Err(TensorError::ComponentExpressionDepthLimitExceeded { .. })
        ));

        let deep_vector = TensorExpr {
            name: "deep".into(),
            dimension: 1,
            indices: vec![TensorIndex::upper("i")],
            components: Some(vec![deep]),
        };
        let scalar =
            TensorExpr::with_components("one", 1, vec![], vec![Expr::from_i64(1)]).unwrap();

        assert!(matches!(
            deep_vector.outer_product(&scalar, "product"),
            Err(TensorError::ComponentExpressionDepthLimitExceeded { .. })
        ));

        let wide = Expr::Add(vec![Expr::from_i64(1); MAX_COMPONENT_EXPRESSION_FANOUT + 1]);
        assert!(matches!(
            TensorExpr::with_components("wide", 1, vec![TensorIndex::upper("i")], vec![wide],),
            Err(TensorError::ComponentExpressionFanoutLimitExceeded { .. })
        ));
    }

    #[test]
    fn outer_product_preserves_the_aggregate_component_expression_limit() {
        fn nested_function(prefix: &str, depth: usize) -> Expr {
            let mut expr = Expr::symbol(format!("{prefix}_leaf"));
            for level in 0..depth {
                expr = Expr::Function(format!("{prefix}_{level}"), vec![expr]);
            }
            expr
        }

        let dimension = 87;
        let left = TensorExpr::with_components(
            "left",
            dimension,
            vec![TensorIndex::upper("i")],
            vec![nested_function("left", 16); dimension],
        )
        .unwrap();
        let right = TensorExpr::with_components(
            "right",
            dimension,
            vec![TensorIndex::lower("j")],
            vec![nested_function("right", 16); dimension],
        )
        .unwrap();

        assert_eq!(
            left.outer_product(&right, "too_many_expression_nodes"),
            Err(TensorError::ComponentExpressionNodeLimitExceeded)
        );
    }

    #[test]
    fn test_metric_tensor_raising_lowering_and_spacetime_interval() {
        // 4D Minkowski Metric eta = diag(-1, 1, 1, 1)
        let eta = MetricTensor::minkowski_4d("eta");
        assert_eq!(eta.dimension, 4);

        // 4-vector v^\mu = (c*t, x, y, z) = (5, 3, 0, 4)
        let v = TensorExpr::with_components(
            "v",
            4,
            vec![TensorIndex::upper("mu")],
            vec![
                Expr::from_i64(5),
                Expr::from_i64(3),
                Expr::from_i64(0),
                Expr::from_i64(4),
            ],
        )
        .unwrap();

        // Lower index: v_\mu = g_{\mu\nu} v^\nu = (-5, 3, 0, 4)
        let v_low = eta.lower_vector(&v).unwrap();
        assert_eq!(v_low.indices[0].variance, IndexVariance::Lower);
        let low_comp = v_low.components.as_ref().unwrap();
        assert_eq!(low_comp[0], Expr::from_i64(-5));
        assert_eq!(low_comp[1], Expr::from_i64(3));
        assert_eq!(low_comp[2], Expr::from_i64(0));
        assert_eq!(low_comp[3], Expr::from_i64(4));

        // Raise index back: v^\mu = (-(-5), 3, 0, 4) = (5, 3, 0, 4)
        let v_up = eta.raise_covector(&v_low).unwrap();
        assert_eq!(v_up.indices[0].variance, IndexVariance::Upper);
        let up_comp = v_up.components.unwrap();
        assert_eq!(up_comp[0], Expr::from_i64(5));
        assert_eq!(up_comp[1], Expr::from_i64(3));

        // Spacetime interval norm squared: ||v||^2 = -5^2 + 3^2 + 0^2 + 4^2 = -25 + 9 + 16 = 0 (lightlike / null vector)
        let norm_sq = eta.norm_squared(&v).unwrap();
        assert_eq!(norm_sq, Expr::from_i64(0));

        // Timelike vector u = (4, 1, 0, 0) -> norm^2 = -16 + 1 = -15
        let u = TensorExpr::with_components(
            "u",
            4,
            vec![TensorIndex::upper("mu")],
            vec![
                Expr::from_i64(4),
                Expr::from_i64(1),
                Expr::from_i64(0),
                Expr::from_i64(0),
            ],
        )
        .unwrap();
        let u_norm = eta.norm_squared(&u).unwrap();
        assert_eq!(u_norm, Expr::from_i64(-15));

        // 3D Euclidean metric
        let euc3 = MetricTensor::euclidean("g", 3).unwrap();
        let e_vec = TensorExpr::with_components(
            "r",
            3,
            vec![TensorIndex::upper("i")],
            vec![Expr::from_i64(1), Expr::from_i64(2), Expr::from_i64(2)],
        )
        .unwrap();
        let euc_norm = euc3.norm_squared(&e_vec).unwrap();
        assert_eq!(euc_norm, Expr::from_i64(9)); // 1 + 4 + 4 = 9

        // Serde roundtrip for MetricTensor
        let eta_wire = serde_json::to_value(&eta).unwrap();
        assert_eq!(
            serde_json::from_value::<MetricTensor>(eta_wire).unwrap(),
            eta
        );
    }

    #[test]
    fn diagonal_metric_refuses_exact_zero_without_unwinding() {
        for zero in [
            Expr::from_i64(0),
            Expr::Rational(BigRational::new(0.into(), 1.into())),
        ] {
            let result = std::panic::catch_unwind(|| {
                MetricTensor::diagonal("singular", vec![Expr::from_i64(1), zero])
            });
            assert!(
                result.is_ok(),
                "exact-zero metric construction must not unwind"
            );
            if let Ok(result) = result {
                assert_eq!(
                    result,
                    Err(TensorError::ZeroDiagonalMetricEntry { index: 1 })
                );
            }
        }
    }

    #[test]
    fn diagonal_metric_inverts_nonunit_exact_entries() {
        let metric = MetricTensor::diagonal(
            "exact",
            vec![
                Expr::from_i64(2),
                Expr::Rational(BigRational::new(2.into(), 3.into())),
            ],
        )
        .expect("nonzero exact diagonal entries are invertible");

        assert_eq!(
            metric.inverse,
            vec![
                Expr::Rational(BigRational::new(1.into(), 2.into())),
                Expr::from_i64(0),
                Expr::from_i64(0),
                Expr::Rational(BigRational::new(3.into(), 2.into())),
            ]
        );
    }
}
