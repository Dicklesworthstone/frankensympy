//! Semantic Term DAG representation for WS04.
//!
//! Subexpression sharing, arena interning, and structural deduplication
//! indexed by typed [`TermId`]. Guarantees acyclicity and identical canonical ID
//! for isomorphic subexpressions.

use crate::{Constant, CoreError, Expr, Symbol};
use fsym_id::TermId;
use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::HashMap;
use std::sync::Arc;

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
}

/// An arena-interned Semantic Term DAG.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermDag {
    nodes: Vec<TermNode>,
    index: HashMap<TermNode, usize>,
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

    /// Interns a [`TermNode`], returning its unique [`TermId`].
    pub fn insert_node(&mut self, node: TermNode) -> TermId {
        if let Some(&raw_idx) = self.index.get(&node) {
            return TermId::new(raw_idx as u64 + 1).unwrap();
        }

        let raw_idx = self.nodes.len();
        self.nodes.push(node.clone());
        self.index.insert(node, raw_idx);
        TermId::new(raw_idx as u64 + 1).unwrap()
    }

    /// Recursively inserts an [`Expr`] into the DAG, returning its root [`TermId`].
    pub fn insert_expr(&mut self, expr: &Expr) -> TermId {
        match expr {
            Expr::Sym(s) => self.insert_node(TermNode::Sym(s.clone())),
            Expr::Integer(n) => self.insert_node(TermNode::Integer(n.clone())),
            Expr::Rational(q) => self.insert_node(TermNode::Rational(q.clone())),
            Expr::Const(c) => self.insert_node(TermNode::Const(*c)),
            Expr::Add(terms) => {
                let ids = terms.iter().map(|t| self.insert_expr(t)).collect();
                self.insert_node(TermNode::Add(ids))
            }
            Expr::Mul(terms) => {
                let ids = terms.iter().map(|t| self.insert_expr(t)).collect();
                self.insert_node(TermNode::Mul(ids))
            }
            Expr::Pow(base, exp) => {
                let b_id = self.insert_expr(base);
                let e_id = self.insert_expr(exp);
                self.insert_node(TermNode::Pow(b_id, e_id))
            }
            Expr::Function(name, args) => {
                let ids = args.iter().map(|a| self.insert_expr(a)).collect();
                self.insert_node(TermNode::Function(name.clone(), ids))
            }
        }
    }

    /// Retrieves a term node by its [`TermId`].
    pub fn get(&self, id: TermId) -> Option<&TermNode> {
        let raw = id.raw().checked_sub(1)? as usize;
        self.nodes.get(raw)
    }

    /// Reconstructs the full tree [`Expr`] from a root [`TermId`].
    pub fn to_expr(&self, root: TermId) -> Result<Expr, CoreError> {
        let node = self.get(root).ok_or_else(|| {
            CoreError::InvalidOperation(format!("TermId {root} not found in DAG"))
        })?;

        match node {
            TermNode::Sym(s) => Ok(Expr::Sym(s.clone())),
            TermNode::Integer(n) => Ok(Expr::Integer(n.clone())),
            TermNode::Rational(q) => Ok(Expr::Rational(q.clone())),
            TermNode::Const(c) => Ok(Expr::Const(*c)),
            TermNode::Add(ids) => {
                let terms = ids
                    .iter()
                    .map(|&id| self.to_expr(id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Add(terms))
            }
            TermNode::Mul(ids) => {
                let terms = ids
                    .iter()
                    .map(|&id| self.to_expr(id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Mul(terms))
            }
            TermNode::Pow(b_id, e_id) => {
                let base = self.to_expr(*b_id)?;
                let exp = self.to_expr(*e_id)?;
                Ok(Expr::Pow(Arc::new(base), Arc::new(exp)))
            }
            TermNode::Function(name, ids) => {
                let args = ids
                    .iter()
                    .map(|&id| self.to_expr(id))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Function(name.clone(), args))
            }
        }
    }

    /// Computes the DAG tree depth for a given node.
    pub fn depth(&self, id: TermId) -> usize {
        let Some(node) = self.get(id) else {
            return 0;
        };
        match node {
            TermNode::Sym(_)
            | TermNode::Integer(_)
            | TermNode::Rational(_)
            | TermNode::Const(_) => 1,
            TermNode::Add(ids) | TermNode::Mul(ids) | TermNode::Function(_, ids) => {
                1 + ids.iter().map(|&i| self.depth(i)).max().unwrap_or(0)
            }
            TermNode::Pow(b, e) => 1 + std::cmp::max(self.depth(*b), self.depth(*e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_dag_interning_and_deduplication() {
        let mut dag = TermDag::new();

        let x = Expr::symbol("x");
        let one = Expr::from_i64(1);
        let expr1 = Expr::Add(vec![x.clone(), one.clone()]);
        let expr2 = Expr::Mul(vec![expr1.clone(), expr1.clone()]);

        let root_id = dag.insert_expr(&expr2);

        // Deduplication: 'x', '1', '(x + 1)', and '(x + 1)*(x + 1)'
        // x = 1, 1 = 2, (x+1) = 3, (x+1)*(x+1) = 4 -> total 4 nodes.
        assert_eq!(dag.len(), 4);

        let reconstructed = dag.to_expr(root_id).unwrap();
        assert_eq!(reconstructed, expr2);
    }

    #[test]
    fn term_dag_depth_calculation() {
        let mut dag = TermDag::new();
        let x = Expr::symbol("x");
        let one = Expr::from_i64(1);
        let add = Expr::Add(vec![x, one]);
        let pow = Expr::Pow(Arc::new(add), Arc::new(Expr::from_i64(2)));

        let root = dag.insert_expr(&pow);
        // Leaf (1) -> Add (2) -> Pow (3)
        assert_eq!(dag.depth(root), 3);
    }
}
