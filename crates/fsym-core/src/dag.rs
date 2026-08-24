//! Semantic Term DAG representation for WS04.
//!
//! Subexpression sharing, arena interning, and structural deduplication
//! indexed by stable content-addressed [`TermId`]. Guarantees acyclicity by construction
//! and identical canonical ID for isomorphic subexpressions.

#![forbid(unsafe_code)]

use crate::{Constant, Expr, Symbol};
use fsym_id::TermId;
use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DagError {
    #[error("Dangling child TermId {0:?} does not exist in DAG")]
    DanglingChild(TermId),
    #[error("Cycle detected at TermId {0:?}")]
    CycleDetected(TermId),
    #[error("Recursion depth limit exceeded ({0})")]
    DepthExceeded(usize),
    #[error("Unknown TermId {0:?}")]
    UnknownId(TermId),
}

/// A node in the deduplicated Semantic Term DAG.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TermNode {
    /// Symbolic variable atom.
    Sym(Symbol),
    /// Arbitrary precision integer literal.
    Integer(BigInt),
    /// Exact rational number.
    Rational(BigRational),
    /// Named mathematical constant.
    Const(Constant),
    /// N-ary addition of interned child terms.
    Add(Vec<TermId>),
    /// N-ary multiplication of interned child terms.
    Mul(Vec<TermId>),
    /// Power expression (base, exponent).
    Pow(TermId, TermId),
    /// Named function application with child term arguments.
    Function(String, Vec<TermId>),
    /// Scoped lambda binder.
    Lambda(Vec<Symbol>, TermId),
}

/// Computes a stable, deterministic content-addressed [`TermId`] from a [`TermNode`].
/// Independent of arena allocation order or pointer addresses.
pub fn compute_term_id(node: &TermNode) -> TermId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.term.v1:");
    match node {
        TermNode::Sym(s) => {
            hasher.update(b"sym:");
            hasher.update(s.name.as_bytes());
        }
        TermNode::Integer(n) => {
            hasher.update(b"int:");
            hasher.update(n.to_string().as_bytes());
        }
        TermNode::Rational(q) => {
            hasher.update(b"rat:");
            hasher.update(q.numer().to_string().as_bytes());
            hasher.update(b"/");
            hasher.update(q.denom().to_string().as_bytes());
        }
        TermNode::Const(c) => {
            hasher.update(b"const:");
            hasher.update(c.to_string().as_bytes());
        }
        TermNode::Add(ids) => {
            hasher.update(b"add:");
            for id in ids {
                hasher.update(&id.raw().to_le_bytes());
            }
        }
        TermNode::Mul(ids) => {
            hasher.update(b"mul:");
            for id in ids {
                hasher.update(&id.raw().to_le_bytes());
            }
        }
        TermNode::Pow(base, exp) => {
            hasher.update(b"pow:");
            hasher.update(&base.raw().to_le_bytes());
            hasher.update(&exp.raw().to_le_bytes());
        }
        TermNode::Function(name, ids) => {
            hasher.update(b"func:");
            hasher.update(name.as_bytes());
            for id in ids {
                hasher.update(&id.raw().to_le_bytes());
            }
        }
        TermNode::Lambda(params, body) => {
            hasher.update(b"lambda:");
            for p in params {
                hasher.update(p.name.as_bytes());
                hasher.update(b",");
            }
            hasher.update(&body.raw().to_le_bytes());
        }
    }
    let hash_bytes = hasher.finalize();
    let mut raw_bytes = [0u8; 8];
    raw_bytes.copy_from_slice(&hash_bytes.as_bytes()[0..8]);
    let raw = u64::from_le_bytes(raw_bytes);
    let non_zero_raw = if raw == 0 { 1 } else { raw };
    TermId::new(non_zero_raw).unwrap()
}

/// An arena-interned Semantic Term DAG with stable identity and acyclicity invariants.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermDag {
    nodes: HashMap<TermId, TermNode>,
}

impl TermDag {
    /// Creates an empty Term DAG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct interned nodes in the DAG.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the DAG contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Interns a [`TermNode`], returning its content-addressed [`TermId`].
    /// Fails closed if any child [`TermId`] is not already present in the DAG.
    pub fn insert_node(&mut self, node: TermNode) -> Result<TermId, DagError> {
        // Enforce acyclicity: all child IDs must already be interned in this DAG.
        match &node {
            TermNode::Sym(_)
            | TermNode::Integer(_)
            | TermNode::Rational(_)
            | TermNode::Const(_) => {}
            TermNode::Add(ids) | TermNode::Mul(ids) | TermNode::Function(_, ids) => {
                for id in ids {
                    if !self.nodes.contains_key(id) {
                        return Err(DagError::DanglingChild(*id));
                    }
                }
            }
            TermNode::Pow(b, e) => {
                if !self.nodes.contains_key(b) {
                    return Err(DagError::DanglingChild(*b));
                }
                if !self.nodes.contains_key(e) {
                    return Err(DagError::DanglingChild(*e));
                }
            }
            TermNode::Lambda(_, body) => {
                if !self.nodes.contains_key(body) {
                    return Err(DagError::DanglingChild(*body));
                }
            }
        }

        let term_id = compute_term_id(&node);
        self.nodes.entry(term_id).or_insert(node);
        Ok(term_id)
    }

    /// Recursively inserts an [`Expr`] into the DAG, returning its root [`TermId`].
    pub fn insert_expr(&mut self, expr: &Expr) -> TermId {
        match expr {
            Expr::Sym(s) => self.insert_node(TermNode::Sym(s.clone())).unwrap(),
            Expr::Integer(n) => self.insert_node(TermNode::Integer(n.clone())).unwrap(),
            Expr::Rational(q) => self.insert_node(TermNode::Rational(q.clone())).unwrap(),
            Expr::Const(c) => self.insert_node(TermNode::Const(*c)).unwrap(),
            Expr::Add(terms) => {
                let ids = terms.iter().map(|t| self.insert_expr(t)).collect();
                self.insert_node(TermNode::Add(ids)).unwrap()
            }
            Expr::Mul(terms) => {
                let ids = terms.iter().map(|t| self.insert_expr(t)).collect();
                self.insert_node(TermNode::Mul(ids)).unwrap()
            }
            Expr::Pow(base, exp) => {
                let b_id = self.insert_expr(base);
                let e_id = self.insert_expr(exp);
                self.insert_node(TermNode::Pow(b_id, e_id)).unwrap()
            }
            Expr::Function(name, args) => {
                let ids = args.iter().map(|a| self.insert_expr(a)).collect();
                self.insert_node(TermNode::Function(name.clone(), ids))
                    .unwrap()
            }
        }
    }

    /// Retrieves a term node by its [`TermId`].
    pub fn get(&self, id: TermId) -> Option<&TermNode> {
        self.nodes.get(&id)
    }

    /// Computes the tree depth of an interned term with cycle detection and bounded recursion.
    pub fn depth(&self, id: TermId) -> Result<usize, DagError> {
        let mut visited = HashSet::new();
        self.depth_internal(id, &mut visited, 0, 512)
    }

    fn depth_internal(
        &self,
        id: TermId,
        visited: &mut HashSet<TermId>,
        current_depth: usize,
        max_depth: usize,
    ) -> Result<usize, DagError> {
        if current_depth > max_depth {
            return Err(DagError::DepthExceeded(max_depth));
        }
        if !visited.insert(id) {
            return Err(DagError::CycleDetected(id));
        }

        let node = self.get(id).ok_or(DagError::UnknownId(id))?;
        let d = match node {
            TermNode::Sym(_)
            | TermNode::Integer(_)
            | TermNode::Rational(_)
            | TermNode::Const(_) => Ok(1),
            TermNode::Add(ids) | TermNode::Mul(ids) | TermNode::Function(_, ids) => {
                let mut max_child = 0;
                for &child_id in ids {
                    let cd =
                        self.depth_internal(child_id, visited, current_depth + 1, max_depth)?;
                    max_child = max_child.max(cd);
                }
                Ok(1 + max_child)
            }
            TermNode::Pow(b, e) => {
                let b_d = self.depth_internal(*b, visited, current_depth + 1, max_depth)?;
                let e_d = self.depth_internal(*e, visited, current_depth + 1, max_depth)?;
                Ok(1 + b_d.max(e_d))
            }
            TermNode::Lambda(_, body) => {
                let b_d = self.depth_internal(*body, visited, current_depth + 1, max_depth)?;
                Ok(1 + b_d)
            }
        };

        visited.remove(&id);
        d
    }

    /// Reconstructs a full [`Expr`] AST from an interned [`TermId`] with depth bounds.
    pub fn to_expr(&self, id: TermId) -> Result<Expr, DagError> {
        let mut visited = HashSet::new();
        self.to_expr_internal(id, &mut visited, 0, 512)
    }

    fn to_expr_internal(
        &self,
        id: TermId,
        visited: &mut HashSet<TermId>,
        current_depth: usize,
        max_depth: usize,
    ) -> Result<Expr, DagError> {
        if current_depth > max_depth {
            return Err(DagError::DepthExceeded(max_depth));
        }
        if !visited.insert(id) {
            return Err(DagError::CycleDetected(id));
        }

        let node = self.get(id).ok_or(DagError::UnknownId(id))?;
        let res = match node {
            TermNode::Sym(s) => Ok(Expr::Sym(s.clone())),
            TermNode::Integer(n) => Ok(Expr::Integer(n.clone())),
            TermNode::Rational(q) => Ok(Expr::Rational(q.clone())),
            TermNode::Const(c) => Ok(Expr::Const(*c)),
            TermNode::Add(ids) => {
                let mut terms = Vec::with_capacity(ids.len());
                for &cid in ids {
                    terms.push(self.to_expr_internal(
                        cid,
                        visited,
                        current_depth + 1,
                        max_depth,
                    )?);
                }
                Ok(Expr::Add(terms))
            }
            TermNode::Mul(ids) => {
                let mut terms = Vec::with_capacity(ids.len());
                for &cid in ids {
                    terms.push(self.to_expr_internal(
                        cid,
                        visited,
                        current_depth + 1,
                        max_depth,
                    )?);
                }
                Ok(Expr::Mul(terms))
            }
            TermNode::Pow(b, e) => {
                let base = self.to_expr_internal(*b, visited, current_depth + 1, max_depth)?;
                let exp = self.to_expr_internal(*e, visited, current_depth + 1, max_depth)?;
                Ok(Expr::Pow(Arc::new(base), Arc::new(exp)))
            }
            TermNode::Function(name, ids) => {
                let mut args = Vec::with_capacity(ids.len());
                for &cid in ids {
                    args.push(self.to_expr_internal(cid, visited, current_depth + 1, max_depth)?);
                }
                Ok(Expr::Function(name.clone(), args))
            }
            TermNode::Lambda(_params, body) => {
                // Return body expression for unextended surface syntax
                self.to_expr_internal(*body, visited, current_depth + 1, max_depth)
            }
        };

        visited.remove(&id);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_id_is_stable_and_order_independent() {
        let mut dag1 = TermDag::new();
        let x_id1 = dag1.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let y_id1 = dag1.insert_node(TermNode::Sym(Symbol::new("y"))).unwrap();

        let mut dag2 = TermDag::new();
        // Insert in reverse order
        let y_id2 = dag2.insert_node(TermNode::Sym(Symbol::new("y"))).unwrap();
        let x_id2 = dag2.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();

        assert_eq!(x_id1, x_id2);
        assert_eq!(y_id1, y_id2);
    }

    #[test]
    fn dangling_child_is_rejected_at_insertion() {
        let mut dag = TermDag::new();
        let fake_child = TermId::new(9999).unwrap();
        let err = dag
            .insert_node(TermNode::Add(vec![fake_child]))
            .unwrap_err();
        assert_eq!(err, DagError::DanglingChild(fake_child));
    }

    #[test]
    fn deduplication_and_round_trip() {
        let mut dag = TermDag::new();
        let expr = Expr::Add(vec![
            Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
            Expr::Mul(vec![Expr::symbol("x"), Expr::symbol("y")]),
        ]);

        let root = dag.insert_expr(&expr);
        assert_eq!(dag.to_expr(root).unwrap(), expr);
        assert_eq!(dag.depth(root).unwrap(), 3);
    }
}
