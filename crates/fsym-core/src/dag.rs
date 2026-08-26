//! Provisional Semantic Term DAG representation for WS04.
//!
//! Subexpression sharing, arena interning, and structural deduplication
//! indexed by stable content-addressed [`TermId`]. Child-before-parent insertion
//! guarantees acyclicity. Domain/sort/context typing and alpha-normalized binders
//! are not implemented yet, so arbitrary `Lambda` surface forms remain opaque
//! functions when lowering from [`Expr`].

#![forbid(unsafe_code)]

use crate::{BigInt, BigRational};
use crate::{Constant, Expr, Symbol};
use fsym_id::TermId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;

// Fixed, cross-platform accounting charges. They deliberately upper-bound the
// current Rust layouts on supported targets without making refusal behavior
// depend on pointer width or compiler enum layout.
const TERM_ID_SLOT_CHARGE_BYTES: usize = 8;
const SYMBOL_SLOT_CHARGE_BYTES: usize = 24;
const LIFTED_EXPR_SLOT_CHARGE_BYTES: usize = 64;
const LAMBDA_SURFACE_NAME_CHARGE_BYTES: usize = "Lambda".len();

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
    #[error("Hash collision detected at TermId {0:?}")]
    HashCollision(TermId),
    #[error("Canonical term digest mapped to the reserved zero TermId")]
    ZeroDigest,
    #[error("Term payload length cannot be represented canonically")]
    PayloadLengthOverflow,
    #[error("DAG node limit exceeded ({0})")]
    NodeLimitExceeded(usize),
    #[error("DAG traversal limit exceeded ({0})")]
    TraversalLimitExceeded(usize),
    #[error("DAG expansion limit exceeded ({0})")]
    ExpansionLimitExceeded(usize),
    #[error("Term arity limit exceeded ({0})")]
    ArityLimitExceeded(usize),
    #[error("Term payload byte limit exceeded ({0})")]
    PayloadLimitExceeded(usize),
    #[error("Aggregate DAG payload byte limit exceeded ({0})")]
    TotalPayloadLimitExceeded(usize),
    #[error("Term numeric payload bit limit exceeded ({0})")]
    NumericPayloadLimitExceeded(u64),
    #[error("DAG allocation failed")]
    AllocationFailure,
}

/// Independent admission limits for DAG construction and lifting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DagLimits {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_traversal_nodes: usize,
    pub max_expanded_nodes: usize,
    pub max_arity: usize,
    pub max_payload_bytes: usize,
    pub max_total_payload_bytes: usize,
    pub max_numeric_bits: u64,
}

impl Default for DagLimits {
    fn default() -> Self {
        Self {
            max_depth: 512,
            max_nodes: 100_000,
            max_traversal_nodes: 100_000,
            max_expanded_nodes: 100_000,
            max_arity: 100_000,
            max_payload_bytes: crate::canonical::MAX_SERIALIZED_BYTES,
            max_total_payload_bytes: 8 * crate::canonical::MAX_SERIALIZED_BYTES,
            max_numeric_bits: 8 * 1024 * 1024,
        }
    }
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
    /// Name-preserving Lambda placeholder. This variant is not yet an
    /// alpha-normalized semantic binder and is not produced by `insert_expr`.
    Lambda(Vec<Symbol>, TermId),
}

fn hash_len(hasher: &mut blake3::Hasher, len: usize) -> Result<(), DagError> {
    let len = u64::try_from(len).map_err(|_| DagError::PayloadLengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    Ok(())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<(), DagError> {
    hash_len(hasher, bytes.len())?;
    hasher.update(bytes);
    Ok(())
}

fn hash_ids(hasher: &mut blake3::Hasher, ids: &[TermId]) -> Result<(), DagError> {
    hash_len(hasher, ids.len())?;
    for id in ids {
        hasher.update(&id.raw().to_le_bytes());
    }
    Ok(())
}

fn hash_integer(hasher: &mut blake3::Hasher, value: &BigInt) -> Result<(), DagError> {
    hasher.update(&[u8::from(value.is_negative())]);
    hash_bytes(hasher, &value.to_bytes_le())
}

fn compute_term_id_unchecked(node: &TermNode) -> Result<TermId, DagError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.term.v2\0");
    match node {
        TermNode::Sym(s) => {
            hasher.update(&[0]);
            hash_bytes(&mut hasher, s.name.as_bytes())?;
        }
        TermNode::Integer(n) => {
            hasher.update(&[1]);
            hash_integer(&mut hasher, n)?;
        }
        TermNode::Rational(q) => {
            hasher.update(&[2]);
            hash_integer(&mut hasher, q.numer())?;
            hash_integer(&mut hasher, q.denom())?;
        }
        TermNode::Const(c) => {
            hasher.update(&[3]);
            hasher.update(&[match c {
                Constant::Pi => 0,
                Constant::E => 1,
                Constant::I => 2,
                Constant::Infinity => 3,
                Constant::NegativeInfinity => 4,
                Constant::ComplexInfinity => 5,
                Constant::NaN => 6,
            }]);
        }
        TermNode::Add(ids) => {
            hasher.update(&[4]);
            hash_ids(&mut hasher, ids)?;
        }
        TermNode::Mul(ids) => {
            hasher.update(&[5]);
            hash_ids(&mut hasher, ids)?;
        }
        TermNode::Pow(base, exp) => {
            hasher.update(&[6]);
            hasher.update(&base.raw().to_le_bytes());
            hasher.update(&exp.raw().to_le_bytes());
        }
        TermNode::Function(name, ids) => {
            hasher.update(&[7]);
            hash_bytes(&mut hasher, name.as_bytes())?;
            hash_ids(&mut hasher, ids)?;
        }
        TermNode::Lambda(params, body) => {
            hasher.update(&[8]);
            hash_len(&mut hasher, params.len())?;
            for parameter in params {
                hash_bytes(&mut hasher, parameter.name.as_bytes())?;
            }
            hasher.update(&body.raw().to_le_bytes());
        }
    }
    let mut digest = hasher.finalize_xof();
    let mut raw_bytes = [0u8; 8];
    digest.fill(&mut raw_bytes);
    let raw = u64::from_le_bytes(raw_bytes);
    TermId::new(raw).map_err(|_| DagError::ZeroDigest)
}

fn node_arity(node: &TermNode) -> Result<usize, DagError> {
    Ok(match node {
        TermNode::Add(ids) | TermNode::Mul(ids) | TermNode::Function(_, ids) => ids.len(),
        TermNode::Pow(..) => 2,
        TermNode::Lambda(parameters, _) => parameters
            .len()
            .checked_add(1)
            .ok_or(DagError::PayloadLengthOverflow)?,
        TermNode::Sym(_) | TermNode::Integer(_) | TermNode::Rational(_) | TermNode::Const(_) => 0,
    })
}

fn numeric_payload_bytes(value: &BigInt) -> Result<usize, DagError> {
    usize::try_from(value.bits().div_ceil(8)).map_err(|_| DagError::PayloadLengthOverflow)
}

fn sequence_storage_bytes(len: usize, slot_charge_bytes: usize) -> Result<usize, DagError> {
    len.checked_mul(slot_charge_bytes)
        .ok_or(DagError::PayloadLengthOverflow)
}

fn add_payload_bytes(lhs: usize, rhs: usize) -> Result<usize, DagError> {
    lhs.checked_add(rhs).ok_or(DagError::PayloadLengthOverflow)
}

fn node_payload_bytes(node: &TermNode) -> Result<usize, DagError> {
    match node {
        TermNode::Sym(symbol) => Ok(symbol.name.len()),
        TermNode::Integer(value) => numeric_payload_bytes(value),
        TermNode::Rational(value) => add_payload_bytes(
            numeric_payload_bytes(value.numer())?,
            numeric_payload_bytes(value.denom())?,
        ),
        TermNode::Add(ids) | TermNode::Mul(ids) => {
            sequence_storage_bytes(ids.len(), TERM_ID_SLOT_CHARGE_BYTES)
        }
        TermNode::Function(name, ids) => add_payload_bytes(
            name.len(),
            sequence_storage_bytes(ids.len(), TERM_ID_SLOT_CHARGE_BYTES)?,
        ),
        TermNode::Lambda(parameters, _) => {
            let parameter_storage =
                sequence_storage_bytes(parameters.len(), SYMBOL_SLOT_CHARGE_BYTES)?;
            parameters
                .iter()
                .try_fold(parameter_storage, |total, parameter| {
                    total
                        .checked_add(parameter.name.len())
                        .ok_or(DagError::PayloadLengthOverflow)
                })
        }
        TermNode::Pow(..) | TermNode::Const(_) => Ok(0),
    }
}

fn lifted_node_payload_bytes(node: &TermNode) -> Result<usize, DagError> {
    match node {
        TermNode::Sym(symbol) => Ok(symbol.name.len()),
        TermNode::Integer(value) => numeric_payload_bytes(value),
        TermNode::Rational(value) => add_payload_bytes(
            numeric_payload_bytes(value.numer())?,
            numeric_payload_bytes(value.denom())?,
        ),
        TermNode::Add(ids) | TermNode::Mul(ids) => {
            sequence_storage_bytes(ids.len(), LIFTED_EXPR_SLOT_CHARGE_BYTES)
        }
        TermNode::Pow(..) => sequence_storage_bytes(2, LIFTED_EXPR_SLOT_CHARGE_BYTES),
        TermNode::Function(name, ids) => add_payload_bytes(
            name.len(),
            sequence_storage_bytes(ids.len(), LIFTED_EXPR_SLOT_CHARGE_BYTES)?,
        ),
        TermNode::Lambda(parameters, _) => {
            let output_arity = parameters
                .len()
                .checked_add(1)
                .ok_or(DagError::PayloadLengthOverflow)?;
            let expression_storage = add_payload_bytes(
                sequence_storage_bytes(output_arity, LIFTED_EXPR_SLOT_CHARGE_BYTES)?,
                LAMBDA_SURFACE_NAME_CHARGE_BYTES,
            )?;
            parameters
                .iter()
                .try_fold(expression_storage, |total, parameter| {
                    total
                        .checked_add(parameter.name.len())
                        .ok_or(DagError::PayloadLengthOverflow)
                })
        }
        TermNode::Const(_) => Ok(0),
    }
}

fn charge_total_payload(total: &mut usize, amount: usize, limit: usize) -> Result<(), DagError> {
    let next = total
        .checked_add(amount)
        .ok_or(DagError::PayloadLengthOverflow)?;
    if next > limit {
        return Err(DagError::TotalPayloadLimitExceeded(limit));
    }
    *total = next;
    Ok(())
}

fn validate_node_limits(node: &TermNode, limits: DagLimits) -> Result<(), DagError> {
    if node_arity(node)? > limits.max_arity {
        return Err(DagError::ArityLimitExceeded(limits.max_arity));
    }
    let payload_bytes = node_payload_bytes(node)?;
    if payload_bytes > limits.max_payload_bytes {
        return Err(DagError::PayloadLimitExceeded(limits.max_payload_bytes));
    }
    let numeric_bits = match node {
        TermNode::Integer(value) => value.bits(),
        TermNode::Rational(value) => value.numer().bits().max(value.denom().bits()),
        _ => 0,
    };
    if numeric_bits > limits.max_numeric_bits {
        return Err(DagError::NumericPayloadLimitExceeded(
            limits.max_numeric_bits,
        ));
    }
    Ok(())
}

fn validate_expr_local_limits(expr: &Expr, limits: DagLimits) -> Result<(), DagError> {
    let arity = match expr {
        Expr::Add(terms) | Expr::Mul(terms) | Expr::Function(_, terms) => terms.len(),
        Expr::Pow(..) => 2,
        Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => 0,
    };
    if arity > limits.max_arity {
        return Err(DagError::ArityLimitExceeded(limits.max_arity));
    }
    let payload_bytes = match expr {
        Expr::Sym(symbol) => symbol.name.len(),
        Expr::Integer(value) => numeric_payload_bytes(value)?,
        Expr::Rational(value) => add_payload_bytes(
            numeric_payload_bytes(value.numer())?,
            numeric_payload_bytes(value.denom())?,
        )?,
        Expr::Add(terms) | Expr::Mul(terms) => {
            sequence_storage_bytes(terms.len(), TERM_ID_SLOT_CHARGE_BYTES)?
        }
        Expr::Function(name, arguments) => add_payload_bytes(
            name.len(),
            sequence_storage_bytes(arguments.len(), TERM_ID_SLOT_CHARGE_BYTES)?,
        )?,
        Expr::Const(_) | Expr::Pow(..) => 0,
    };
    if payload_bytes > limits.max_payload_bytes {
        return Err(DagError::PayloadLimitExceeded(limits.max_payload_bytes));
    }
    let numeric_bits = match expr {
        Expr::Integer(value) => value.bits(),
        Expr::Rational(value) => value.numer().bits().max(value.denom().bits()),
        Expr::Sym(_)
        | Expr::Const(_)
        | Expr::Add(_)
        | Expr::Mul(_)
        | Expr::Pow(..)
        | Expr::Function(..) => 0,
    };
    if numeric_bits > limits.max_numeric_bits {
        return Err(DagError::NumericPayloadLimitExceeded(
            limits.max_numeric_bits,
        ));
    }
    Ok(())
}

fn next_depth(current: usize, limit: usize) -> Result<usize, DagError> {
    current.checked_add(1).ok_or(DagError::DepthExceeded(limit))
}

/// Computes a stable content-addressed [`TermId`] from an unambiguous,
/// length-framed canonical preimage under default payload and arity limits.
/// Independent of arena allocation order and pointer addresses.
pub fn compute_term_id(node: &TermNode) -> Result<TermId, DagError> {
    compute_term_id_with_limits(node, DagLimits::default())
}

/// Computes a stable content ID under caller-provided admission limits.
pub fn compute_term_id_with_limits(node: &TermNode, limits: DagLimits) -> Result<TermId, DagError> {
    validate_node_limits(node, limits)?;
    compute_term_id_unchecked(node)
}

/// An arena-interned Semantic Term DAG with stable identity and acyclicity invariants.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermDag {
    nodes: HashMap<TermId, TermNode>,
    depths: HashMap<TermId, usize>,
    total_payload_bytes: usize,
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

    /// Total variable-size payload bytes owned by distinct interned nodes.
    pub fn total_payload_bytes(&self) -> usize {
        self.total_payload_bytes
    }

    /// Interns a [`TermNode`], returning its content-addressed [`TermId`].
    /// Fails closed if any child [`TermId`] is not already present in the DAG.
    pub fn insert_node(&mut self, node: TermNode) -> Result<TermId, DagError> {
        self.insert_node_with_limits(node, DagLimits::default())
    }

    /// Interns one node under caller-provided admission limits.
    pub fn insert_node_with_limits(
        &mut self,
        node: TermNode,
        limits: DagLimits,
    ) -> Result<TermId, DagError> {
        validate_node_limits(&node, limits)?;
        self.validate_child_links(&node)?;
        let node_depth = self.prospective_node_depth(&node, limits)?;
        let term_id = compute_term_id_unchecked(&node)?;
        let payload_bytes = node_payload_bytes(&node)?;
        self.intern_prehashed_node(node, term_id, node_depth, payload_bytes, limits)
    }

    fn validate_child_links(&self, node: &TermNode) -> Result<(), DagError> {
        // Enforce acyclicity: all child IDs must already be interned in this DAG.
        match node {
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
        Ok(())
    }

    fn prospective_node_depth(
        &self,
        node: &TermNode,
        limits: DagLimits,
    ) -> Result<usize, DagError> {
        let child_depth = |child: &TermId| {
            self.depths
                .get(child)
                .copied()
                .ok_or(DagError::UnknownId(*child))
        };
        let max_child_depth = match node {
            TermNode::Add(ids) | TermNode::Mul(ids) | TermNode::Function(_, ids) => {
                let mut max_depth = 0;
                for child in ids {
                    max_depth = max_depth.max(child_depth(child)?);
                }
                max_depth
            }
            TermNode::Pow(base, exponent) => child_depth(base)?.max(child_depth(exponent)?),
            TermNode::Lambda(_, body) => child_depth(body)?,
            TermNode::Sym(_)
            | TermNode::Integer(_)
            | TermNode::Rational(_)
            | TermNode::Const(_) => 0,
        };
        let node_depth = max_child_depth
            .checked_add(1)
            .ok_or(DagError::DepthExceeded(limits.max_depth))?;
        if node_depth.saturating_sub(1) > limits.max_depth {
            return Err(DagError::DepthExceeded(limits.max_depth));
        }
        Ok(node_depth)
    }

    fn intern_prehashed_node(
        &mut self,
        node: TermNode,
        term_id: TermId,
        node_depth: usize,
        payload_bytes: usize,
        limits: DagLimits,
    ) -> Result<TermId, DagError> {
        if let Some(existing) = self.nodes.get(&term_id) {
            if existing != &node {
                return Err(DagError::HashCollision(term_id));
            }
        } else {
            if self.nodes.len() >= limits.max_nodes {
                return Err(DagError::NodeLimitExceeded(limits.max_nodes));
            }
            let mut next_total_payload = self.total_payload_bytes;
            charge_total_payload(
                &mut next_total_payload,
                payload_bytes,
                limits.max_total_payload_bytes,
            )?;
            self.nodes
                .try_reserve(1)
                .map_err(|_| DagError::AllocationFailure)?;
            self.depths
                .try_reserve(1)
                .map_err(|_| DagError::AllocationFailure)?;
            self.nodes.insert(term_id, node);
            self.depths.insert(term_id, node_depth);
            self.total_payload_bytes = next_total_payload;
        }
        Ok(term_id)
    }

    /// Recursively inserts an [`Expr`] under default traversal and arena limits.
    pub fn insert_expr(&mut self, expr: &Expr) -> Result<TermId, DagError> {
        self.insert_expr_with_limits(expr, DagLimits::default())
    }

    /// Recursively inserts an [`Expr`] under caller-provided limits.
    pub fn insert_expr_with_limits(
        &mut self,
        expr: &Expr,
        limits: DagLimits,
    ) -> Result<TermId, DagError> {
        let mut traversed = 0_usize;
        let mut inserted = Vec::new();
        match self.insert_expr_internal(expr, 0, limits, &mut traversed, &mut inserted) {
            Ok(root) => Ok(root),
            Err(error) => {
                for id in inserted.into_iter().rev() {
                    let node = self.nodes.remove(&id).ok_or(DagError::UnknownId(id))?;
                    self.depths.remove(&id).ok_or(DagError::UnknownId(id))?;
                    let payload_bytes = node_payload_bytes(&node)?;
                    self.total_payload_bytes = self
                        .total_payload_bytes
                        .checked_sub(payload_bytes)
                        .ok_or(DagError::PayloadLengthOverflow)?;
                }
                Err(error)
            }
        }
    }

    fn insert_expr_internal(
        &mut self,
        expr: &Expr,
        depth: usize,
        limits: DagLimits,
        traversed: &mut usize,
        inserted: &mut Vec<TermId>,
    ) -> Result<TermId, DagError> {
        if depth > limits.max_depth {
            return Err(DagError::DepthExceeded(limits.max_depth));
        }
        if *traversed >= limits.max_traversal_nodes {
            return Err(DagError::TraversalLimitExceeded(limits.max_traversal_nodes));
        }
        *traversed += 1;
        validate_expr_local_limits(expr, limits)?;

        match expr {
            Expr::Sym(s) => self.insert_node_tracking(TermNode::Sym(s.clone()), limits, inserted),
            Expr::Integer(n) => {
                self.insert_node_tracking(TermNode::Integer(n.clone()), limits, inserted)
            }
            Expr::Rational(q) => {
                self.insert_node_tracking(TermNode::Rational(q.clone()), limits, inserted)
            }
            Expr::Const(c) => self.insert_node_tracking(TermNode::Const(*c), limits, inserted),
            Expr::Add(terms) => {
                let ids = self.insert_expr_children(terms, depth, limits, traversed, inserted)?;
                self.insert_node_tracking(TermNode::Add(ids), limits, inserted)
            }
            Expr::Mul(terms) => {
                let ids = self.insert_expr_children(terms, depth, limits, traversed, inserted)?;
                self.insert_node_tracking(TermNode::Mul(ids), limits, inserted)
            }
            Expr::Pow(base, exp) => {
                let child_depth = next_depth(depth, limits.max_depth)?;
                let base_id =
                    self.insert_expr_internal(base, child_depth, limits, traversed, inserted)?;
                let exponent_id =
                    self.insert_expr_internal(exp, child_depth, limits, traversed, inserted)?;
                self.insert_node_tracking(TermNode::Pow(base_id, exponent_id), limits, inserted)
            }
            Expr::Function(name, args) => {
                // Binders remain opaque until the kernel has a capture-avoiding,
                // alpha-normalized representation. Name-based lowering would
                // conflate surface spelling with semantic binding identity.
                let ids = self.insert_expr_children(args, depth, limits, traversed, inserted)?;
                self.insert_node_tracking(TermNode::Function(name.clone(), ids), limits, inserted)
            }
        }
    }

    fn insert_node_tracking(
        &mut self,
        node: TermNode,
        limits: DagLimits,
        inserted: &mut Vec<TermId>,
    ) -> Result<TermId, DagError> {
        validate_node_limits(&node, limits)?;
        self.validate_child_links(&node)?;
        let node_depth = self.prospective_node_depth(&node, limits)?;
        let expected_id = compute_term_id_unchecked(&node)?;
        let payload_bytes = node_payload_bytes(&node)?;
        let is_new = !self.nodes.contains_key(&expected_id);
        if is_new {
            inserted
                .try_reserve(1)
                .map_err(|_| DagError::AllocationFailure)?;
        }
        let actual_id =
            self.intern_prehashed_node(node, expected_id, node_depth, payload_bytes, limits)?;
        if is_new {
            inserted.push(actual_id);
        }
        Ok(actual_id)
    }

    fn insert_expr_children(
        &mut self,
        children: &[Expr],
        parent_depth: usize,
        limits: DagLimits,
        traversed: &mut usize,
        inserted: &mut Vec<TermId>,
    ) -> Result<Vec<TermId>, DagError> {
        if children.len() > limits.max_arity {
            return Err(DagError::ArityLimitExceeded(limits.max_arity));
        }
        let mut ids = Vec::new();
        ids.try_reserve(children.len())
            .map_err(|_| DagError::AllocationFailure)?;
        let child_depth = if children.is_empty() {
            parent_depth
        } else {
            next_depth(parent_depth, limits.max_depth)?
        };
        for child in children {
            ids.push(self.insert_expr_internal(child, child_depth, limits, traversed, inserted)?);
        }
        Ok(ids)
    }

    /// Retrieves a term node by its [`TermId`].
    pub fn get(&self, id: TermId) -> Option<&TermNode> {
        self.nodes.get(&id)
    }

    /// Computes term depth with cycle detection and default limits.
    pub fn depth(&self, id: TermId) -> Result<usize, DagError> {
        self.depth_with_limits(id, DagLimits::default())
    }

    /// Computes term depth with caller-provided limits. Shared subterms are
    /// memoized, so a diamond DAG is visited once per unique node.
    pub fn depth_with_limits(&self, id: TermId, limits: DagLimits) -> Result<usize, DagError> {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        let mut traversed = 0_usize;
        self.depth_internal(id, &mut visiting, &mut memo, &mut traversed, 0, limits)
    }

    fn depth_internal(
        &self,
        id: TermId,
        visiting: &mut HashSet<TermId>,
        memo: &mut HashMap<TermId, usize>,
        traversed: &mut usize,
        current_depth: usize,
        limits: DagLimits,
    ) -> Result<usize, DagError> {
        if current_depth > limits.max_depth {
            return Err(DagError::DepthExceeded(limits.max_depth));
        }
        if let Some(depth) = memo.get(&id) {
            let deepest_level = current_depth
                .checked_add(depth.saturating_sub(1))
                .ok_or(DagError::DepthExceeded(limits.max_depth))?;
            if deepest_level > limits.max_depth {
                return Err(DagError::DepthExceeded(limits.max_depth));
            }
            return Ok(*depth);
        }
        if *traversed >= limits.max_traversal_nodes {
            return Err(DagError::TraversalLimitExceeded(limits.max_traversal_nodes));
        }
        *traversed += 1;
        if visiting.contains(&id) {
            return Err(DagError::CycleDetected(id));
        }
        visiting
            .try_reserve(1)
            .map_err(|_| DagError::AllocationFailure)?;
        visiting.insert(id);

        let node = self.get(id).ok_or(DagError::UnknownId(id))?;
        validate_node_limits(node, limits)?;
        let depth = match node {
            TermNode::Sym(_)
            | TermNode::Integer(_)
            | TermNode::Rational(_)
            | TermNode::Const(_) => 1,
            TermNode::Add(ids) | TermNode::Mul(ids) | TermNode::Function(_, ids) => {
                let mut max_child = 0;
                let child_level = if ids.is_empty() {
                    current_depth
                } else {
                    next_depth(current_depth, limits.max_depth)?
                };
                for &child_id in ids {
                    let child_depth = self.depth_internal(
                        child_id,
                        visiting,
                        memo,
                        traversed,
                        child_level,
                        limits,
                    )?;
                    max_child = max_child.max(child_depth);
                }
                max_child
                    .checked_add(1)
                    .ok_or(DagError::DepthExceeded(limits.max_depth))?
            }
            TermNode::Pow(base, exponent) => {
                let child_level = next_depth(current_depth, limits.max_depth)?;
                let base_depth =
                    self.depth_internal(*base, visiting, memo, traversed, child_level, limits)?;
                let exponent_depth =
                    self.depth_internal(*exponent, visiting, memo, traversed, child_level, limits)?;
                base_depth
                    .max(exponent_depth)
                    .checked_add(1)
                    .ok_or(DagError::DepthExceeded(limits.max_depth))?
            }
            TermNode::Lambda(_, body) => {
                let child_level = next_depth(current_depth, limits.max_depth)?;
                self.depth_internal(*body, visiting, memo, traversed, child_level, limits)?
                    .checked_add(1)
                    .ok_or(DagError::DepthExceeded(limits.max_depth))?
            }
        };

        visiting.remove(&id);
        memo.try_reserve(1)
            .map_err(|_| DagError::AllocationFailure)?;
        memo.insert(id, depth);
        Ok(depth)
    }

    /// Reconstructs a full [`Expr`] tree under default depth and expansion bounds.
    pub fn to_expr(&self, id: TermId) -> Result<Expr, DagError> {
        self.to_expr_with_limits(id, DagLimits::default())
    }

    /// Reconstructs a full [`Expr`] tree under caller-provided limits. Every
    /// duplicated occurrence counts against `max_expanded_nodes` even when the
    /// source DAG shares that term.
    pub fn to_expr_with_limits(&self, id: TermId, limits: DagLimits) -> Result<Expr, DagError> {
        let mut visiting = HashSet::new();
        let mut expanded = 0_usize;
        let mut emitted_payload_bytes = 0_usize;
        self.to_expr_internal(
            id,
            &mut visiting,
            &mut expanded,
            &mut emitted_payload_bytes,
            0,
            limits,
        )
    }

    fn to_expr_internal(
        &self,
        id: TermId,
        visiting: &mut HashSet<TermId>,
        expanded: &mut usize,
        emitted_payload_bytes: &mut usize,
        current_depth: usize,
        limits: DagLimits,
    ) -> Result<Expr, DagError> {
        if current_depth > limits.max_depth {
            return Err(DagError::DepthExceeded(limits.max_depth));
        }
        if *expanded >= limits.max_expanded_nodes {
            return Err(DagError::ExpansionLimitExceeded(limits.max_expanded_nodes));
        }
        *expanded += 1;
        if visiting.contains(&id) {
            return Err(DagError::CycleDetected(id));
        }
        visiting
            .try_reserve(1)
            .map_err(|_| DagError::AllocationFailure)?;
        visiting.insert(id);

        let node = self.get(id).ok_or(DagError::UnknownId(id))?;
        validate_node_limits(node, limits)?;
        charge_total_payload(
            emitted_payload_bytes,
            lifted_node_payload_bytes(node)?,
            limits.max_total_payload_bytes,
        )?;
        let result = match node {
            TermNode::Sym(symbol) => Ok(Expr::Sym(symbol.clone())),
            TermNode::Integer(value) => Ok(Expr::Integer(value.clone())),
            TermNode::Rational(value) => Ok(Expr::Rational(value.clone())),
            TermNode::Const(constant) => Ok(Expr::Const(*constant)),
            TermNode::Add(ids) => {
                let child_level = if ids.is_empty() {
                    current_depth
                } else {
                    next_depth(current_depth, limits.max_depth)?
                };
                let mut terms = Vec::new();
                terms
                    .try_reserve(ids.len())
                    .map_err(|_| DagError::AllocationFailure)?;
                for &child_id in ids {
                    terms.push(self.to_expr_internal(
                        child_id,
                        visiting,
                        expanded,
                        emitted_payload_bytes,
                        child_level,
                        limits,
                    )?);
                }
                Ok(Expr::Add(terms))
            }
            TermNode::Mul(ids) => {
                let child_level = if ids.is_empty() {
                    current_depth
                } else {
                    next_depth(current_depth, limits.max_depth)?
                };
                let mut factors = Vec::new();
                factors
                    .try_reserve(ids.len())
                    .map_err(|_| DagError::AllocationFailure)?;
                for &child_id in ids {
                    factors.push(self.to_expr_internal(
                        child_id,
                        visiting,
                        expanded,
                        emitted_payload_bytes,
                        child_level,
                        limits,
                    )?);
                }
                Ok(Expr::Mul(factors))
            }
            TermNode::Pow(base, exponent) => {
                let child_level = next_depth(current_depth, limits.max_depth)?;
                let base = self.to_expr_internal(
                    *base,
                    visiting,
                    expanded,
                    emitted_payload_bytes,
                    child_level,
                    limits,
                )?;
                let exponent = self.to_expr_internal(
                    *exponent,
                    visiting,
                    expanded,
                    emitted_payload_bytes,
                    child_level,
                    limits,
                )?;
                Ok(Expr::Pow(Arc::new(base), Arc::new(exponent)))
            }
            TermNode::Function(name, ids) => {
                let child_level = if ids.is_empty() {
                    current_depth
                } else {
                    next_depth(current_depth, limits.max_depth)?
                };
                let mut args = Vec::new();
                args.try_reserve(ids.len())
                    .map_err(|_| DagError::AllocationFailure)?;
                for &child_id in ids {
                    args.push(self.to_expr_internal(
                        child_id,
                        visiting,
                        expanded,
                        emitted_payload_bytes,
                        child_level,
                        limits,
                    )?);
                }
                Ok(Expr::Function(name.clone(), args))
            }
            TermNode::Lambda(parameters, body) => {
                let child_level = next_depth(current_depth, limits.max_depth)?;
                let capacity = parameters
                    .len()
                    .checked_add(1)
                    .ok_or(DagError::PayloadLengthOverflow)?;
                let mut args = Vec::new();
                args.try_reserve(capacity)
                    .map_err(|_| DagError::AllocationFailure)?;
                args.extend(
                    parameters
                        .iter()
                        .map(|parameter| Expr::Sym(parameter.clone())),
                );
                let body = self.to_expr_internal(
                    *body,
                    visiting,
                    expanded,
                    emitted_payload_bytes,
                    child_level,
                    limits,
                )?;
                args.push(body);
                Ok(Expr::Function("Lambda".to_string(), args))
            }
        };

        visiting.remove(&id);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_storage_charges_cover_current_native_layouts() {
        assert!(std::mem::size_of::<TermId>() <= TERM_ID_SLOT_CHARGE_BYTES);
        assert!(std::mem::size_of::<Symbol>() <= SYMBOL_SLOT_CHARGE_BYTES);
        assert!(std::mem::size_of::<Expr>() <= LIFTED_EXPR_SLOT_CHARGE_BYTES);
    }

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
    fn canonical_preimage_frames_variable_length_fields() {
        let absorbed_child = TermId::new(u64::from_le_bytes([b'b', 0, 0, 0, 0, 0, 0, 0])).unwrap();
        // The old delimiter-free encoding made these exact preimages equal:
        // name "a" + the child's raw bytes versus one longer function name.
        let with_child = TermNode::Function("a".to_string(), vec![absorbed_child]);
        let absorbed_name = TermNode::Function("ab\0\0\0\0\0\0\0".to_string(), Vec::new());
        assert_ne!(
            compute_term_id(&with_child).unwrap(),
            compute_term_id(&absorbed_name).unwrap()
        );

        // Comma delimiters likewise could not distinguish one parameter name
        // containing a comma from two separate parameter names.
        let body = TermId::new(1).unwrap();
        let one_parameter = TermNode::Lambda(vec![Symbol::new("a,b")], body);
        let two_parameters = TermNode::Lambda(vec![Symbol::new("a"), Symbol::new("b")], body);
        assert_ne!(
            compute_term_id(&one_parameter).unwrap(),
            compute_term_id(&two_parameters).unwrap()
        );

        let tiny_payload = DagLimits {
            max_payload_bytes: 1,
            ..DagLimits::default()
        };
        assert_eq!(
            compute_term_id_with_limits(&TermNode::Sym(Symbol::new("xx")), tiny_payload),
            Err(DagError::PayloadLimitExceeded(1))
        );
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

        let root = dag.insert_expr(&expr).unwrap();
        assert_eq!(dag.to_expr(root).unwrap(), expr);
        assert_eq!(dag.depth(root).unwrap(), 3);
    }

    #[test]
    fn surface_lambda_remains_opaque_until_binding_identity_exists() {
        let expression = Expr::Function(
            "Lambda".to_string(),
            vec![Expr::symbol("x"), Expr::symbol("x")],
        );
        let mut dag = TermDag::new();
        let root = dag.insert_expr(&expression).unwrap();
        assert!(matches!(dag.get(root), Some(TermNode::Function(name, _)) if name == "Lambda"));
        assert_eq!(dag.to_expr(root).unwrap(), expression);
    }

    #[test]
    fn failed_recursive_insert_publishes_no_partial_nodes() {
        let mut dag = TermDag::new();
        let existing = dag
            .insert_node(TermNode::Sym(Symbol::new("existing")))
            .unwrap();
        let before = dag.len();
        let expression = Expr::Add(vec![Expr::symbol("new_a"), Expr::symbol("new_b")]);
        let limits = DagLimits {
            max_traversal_nodes: 2,
            ..DagLimits::default()
        };
        assert_eq!(
            dag.insert_expr_with_limits(&expression, limits),
            Err(DagError::TraversalLimitExceeded(2))
        );
        assert_eq!(dag.len(), before);
        assert!(dag.get(existing).is_some());

        let tiny_payload = DagLimits {
            max_payload_bytes: 1,
            ..DagLimits::default()
        };
        let oversized_function = Expr::Function("xx".to_string(), vec![Expr::symbol("child")]);
        assert_eq!(
            dag.insert_expr_with_limits(&oversized_function, tiny_payload),
            Err(DagError::PayloadLimitExceeded(1))
        );
        assert_eq!(dag.len(), before);
    }

    #[test]
    fn aggregate_payload_limit_is_transactional_and_counts_numeric_storage() {
        let mut dag = TermDag::new();
        let existing = dag.insert_node(TermNode::Sym(Symbol::new("z"))).unwrap();
        assert_eq!(dag.total_payload_bytes(), 1);

        let five_payload_bytes = DagLimits {
            max_total_payload_bytes: 5,
            ..DagLimits::default()
        };
        let expression = Expr::Add(vec![Expr::symbol("aa"), Expr::symbol("bbb")]);
        assert_eq!(
            dag.insert_expr_with_limits(&expression, five_payload_bytes),
            Err(DagError::TotalPayloadLimitExceeded(5))
        );
        assert_eq!(dag.len(), 1, "failed insertion must roll back new leaves");
        assert_eq!(dag.total_payload_bytes(), 1);
        assert!(dag.get(existing).is_some());

        let no_payload = DagLimits {
            max_payload_bytes: 0,
            ..DagLimits::default()
        };
        assert_eq!(
            dag.insert_node_with_limits(TermNode::Integer(BigInt::from(1)), no_payload),
            Err(DagError::PayloadLimitExceeded(0)),
            "numeric magnitude allocations count as variable-size payload"
        );
    }

    #[test]
    fn aggregate_payload_limit_bounds_arena_growth_and_shared_tree_lifting() {
        let three_payload_bytes = DagLimits {
            max_total_payload_bytes: 3,
            ..DagLimits::default()
        };
        let mut bounded = TermDag::new();
        let aa = bounded
            .insert_node_with_limits(TermNode::Sym(Symbol::new("aa")), three_payload_bytes)
            .unwrap();
        assert_eq!(bounded.total_payload_bytes(), 2);
        assert_eq!(
            bounded
                .insert_node_with_limits(TermNode::Sym(Symbol::new("aa")), three_payload_bytes)
                .unwrap(),
            aa,
            "deduplication does not allocate or consume payload budget"
        );
        assert_eq!(
            bounded.insert_node_with_limits(TermNode::Sym(Symbol::new("bb")), three_payload_bytes),
            Err(DagError::TotalPayloadLimitExceeded(3))
        );
        assert_eq!(bounded.total_payload_bytes(), 2);

        let mut shared = TermDag::new();
        let leaf = shared
            .insert_node(TermNode::Sym(Symbol::new("wide")))
            .unwrap();
        let root = shared.insert_node(TermNode::Add(vec![leaf, leaf])).unwrap();
        let seven_output_bytes = DagLimits {
            max_total_payload_bytes: 7,
            ..DagLimits::default()
        };
        assert_eq!(
            shared.to_expr_with_limits(root, seven_output_bytes),
            Err(DagError::TotalPayloadLimitExceeded(7)),
            "duplicated DAG occurrences must each charge their cloned payload"
        );
        assert!(shared.to_expr(root).is_ok());

        let mut lambda_dag = TermDag::new();
        let body = lambda_dag
            .insert_node(TermNode::Sym(Symbol::new("x")))
            .unwrap();
        let lambda = lambda_dag
            .insert_node(TermNode::Lambda(vec![Symbol::new("p")], body))
            .unwrap();
        let omitted_name_would_fit = DagLimits {
            max_total_payload_bytes: 130,
            ..DagLimits::default()
        };
        assert_eq!(
            lambda_dag.to_expr_with_limits(lambda, omitted_name_would_fit),
            Err(DagError::TotalPayloadLimitExceeded(130)),
            "lifting must charge the synthesized `Lambda` function name"
        );
    }

    #[test]
    fn node_and_arity_limits_fail_closed() {
        let mut dag = TermDag::new();
        let one_node = DagLimits {
            max_nodes: 1,
            ..DagLimits::default()
        };
        let x = dag
            .insert_node_with_limits(TermNode::Sym(Symbol::new("x")), one_node)
            .unwrap();
        assert_eq!(
            dag.insert_node_with_limits(TermNode::Sym(Symbol::new("x")), one_node)
                .unwrap(),
            x,
            "deduplication remains available at the arena limit"
        );
        assert_eq!(
            dag.insert_node_with_limits(TermNode::Sym(Symbol::new("y")), one_node),
            Err(DagError::NodeLimitExceeded(1))
        );

        let arity_one = DagLimits {
            max_nodes: 10,
            max_arity: 1,
            ..DagLimits::default()
        };
        assert_eq!(
            dag.insert_node_with_limits(TermNode::Add(vec![x, x]), arity_one),
            Err(DagError::ArityLimitExceeded(1))
        );

        let root_only = DagLimits {
            max_nodes: 10,
            max_depth: 0,
            ..DagLimits::default()
        };
        assert_eq!(
            dag.insert_node_with_limits(TermNode::Add(vec![x]), root_only),
            Err(DagError::DepthExceeded(0)),
            "direct insertion must enforce the resulting DAG depth"
        );
        assert_eq!(dag.len(), 1);
    }

    #[test]
    fn shared_diamond_depth_is_memoized_but_tree_expansion_is_bounded() {
        let mut dag = TermDag::new();
        let mut root = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        for _ in 0..18 {
            root = dag.insert_node(TermNode::Add(vec![root, root])).unwrap();
        }

        assert_eq!(dag.depth(root).unwrap(), 19);
        assert_eq!(
            dag.to_expr(root),
            Err(DagError::ExpansionLimitExceeded(
                DagLimits::default().max_expanded_nodes
            ))
        );
    }

    #[test]
    fn memoized_depth_still_enforces_the_deeper_parent_path() {
        let mut dag = TermDag::new();
        let leaf = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let tail_1 = dag.insert_node(TermNode::Add(vec![leaf])).unwrap();
        let tail_2 = dag.insert_node(TermNode::Add(vec![tail_1])).unwrap();
        let deep_1 = dag.insert_node(TermNode::Add(vec![tail_2])).unwrap();
        let deep_2 = dag.insert_node(TermNode::Add(vec![deep_1])).unwrap();
        let deep_3 = dag.insert_node(TermNode::Add(vec![deep_2])).unwrap();
        // Visit tail_2 first so it is memoized at the shallow root path, then
        // encounter the same height-three tail below three additional parents.
        let root = dag
            .insert_node(TermNode::Add(vec![tail_2, deep_3]))
            .unwrap();
        let limits = DagLimits {
            max_depth: 5,
            ..DagLimits::default()
        };
        assert_eq!(
            dag.depth_with_limits(root, limits),
            Err(DagError::DepthExceeded(5))
        );
    }

    #[test]
    fn concurrent_interning_yields_identical_content_ids() {
        // WS04 acceptance: concurrent interning schedule exploration.
        // Four threads insert the same symbol set under different rotation
        // offsets, producing distinct lock-interleaving schedules. Content
        // identity must hold regardless of schedule; interning deduplicates
        // to exactly one DAG node per distinct term.
        use std::sync::Mutex;

        let dag = Mutex::new(TermDag::new());
        let symbols: Vec<String> = (0..64).map(|i| format!("v{i}")).collect();

        let all_observed: Vec<Vec<(usize, TermId)>> = std::thread::scope(|scope| {
            (0..4)
                .map(|offset| {
                    let dag_ref = &dag;
                    let symbols_ref = &symbols;
                    scope.spawn(move || {
                        let mut observed = Vec::new();
                        for step in 0..symbols_ref.len() {
                            let index = (step + offset) % symbols_ref.len();
                            let id = dag_ref
                                .lock()
                                .expect("interning mutex poisoned")
                                .insert_node(TermNode::Sym(Symbol::new(symbols_ref[index].clone())))
                                .expect("symbol insertion cannot exceed depth");
                            observed.push((index, id));
                        }
                        observed
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("interning thread panicked"))
                .collect()
        });
        // Schedules differ per thread, so compare the content->identity
        // MAPPING rather than visitation order.
        let observed_maps: Vec<std::collections::BTreeMap<usize, TermId>> = all_observed
            .into_iter()
            .map(|thread| thread.into_iter().collect())
            .collect();
        for window in observed_maps.windows(2) {
            assert_eq!(window[0], window[1], "schedules must not affect identity");
        }

        let dag = dag.into_inner().unwrap();
        assert_eq!(dag.len(), symbols.len(), "interning must deduplicate");
    }
}
