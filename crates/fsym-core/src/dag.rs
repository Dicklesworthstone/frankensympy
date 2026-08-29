//! Provisional Semantic Term DAG representation for WS04.
//!
//! Subexpression sharing, arena interning, and structural deduplication
//! indexed by stable content-addressed [`TermId`]. Child-before-parent insertion
//! guarantees acyclicity. Declared [`TermDomain`] is intern identity, distinct
//! from inferred [`Sort`]. Well-formed symbol-parameter `Lambda` surface lowers
//! to name-preserving [`TermNode::Lambda`]; that is not yet alpha-normalized
//! binder identity. Tuple-parameter `Lambda` intern as the same binder node as
//! the multi-argument spelling. Lifting emits the multi-argument surface.

#![forbid(unsafe_code)]

use crate::domain::TermDomain;
use crate::sort::Sort;
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
    #[error("{operation} requires numeric scalar operands, found {actual}")]
    SortMismatch {
        operation: &'static str,
        actual: Sort,
    },
    #[error("term is incompatible with declared domain {domain}: {reason}")]
    DomainIncompatible {
        domain: TermDomain,
        reason: &'static str,
    },
    #[error("malformed binder {name}: {reason}")]
    MalformedBinder {
        name: &'static str,
        reason: &'static str,
    },
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
    /// Name-preserving Lambda binder produced from well-formed symbol-parameter
    /// surface. Parameter names are intern identity; this is not yet
    /// alpha-normalized.
    Lambda(Vec<Symbol>, TermId),
}

impl TermNode {
    /// Returns the conservative intrinsic mathematical sort of this term node.
    pub fn intrinsic_sort(&self) -> crate::sort::Sort {
        match self {
            TermNode::Integer(_) => crate::sort::Sort::Integer,
            TermNode::Rational(_) => crate::sort::Sort::Rational,
            TermNode::Const(c) => match c {
                Constant::Pi | Constant::E | Constant::Infinity | Constant::NegativeInfinity => {
                    crate::sort::Sort::Real
                }
                Constant::I | Constant::ComplexInfinity => crate::sort::Sort::Complex,
                Constant::NaN => crate::sort::Sort::Scalar,
            },
            TermNode::Sym(_) => crate::sort::Sort::Scalar,
            TermNode::Add(_) | TermNode::Mul(_) | TermNode::Pow(..) | TermNode::Function(..) => {
                crate::sort::Sort::Scalar
            }
            TermNode::Lambda(params, _) => crate::sort::Sort::Function {
                dom: vec![crate::sort::Sort::Scalar; params.len()],
                codom: Box::new(crate::sort::Sort::Scalar),
            },
        }
    }
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

fn hash_term_preimage(node: &TermNode, domain: TermDomain) -> Result<blake3::Hasher, DagError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.term.v3\0");
    hasher.update(&[domain.tag()]);
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
    Ok(hasher)
}

/// Full 32-byte BLAKE3 digest of the canonical term preimage in the
/// default [`TermDomain::Expression`] intern universe.
pub fn compute_term_digest(node: &TermNode) -> Result<[u8; 32], DagError> {
    compute_term_digest_in_domain(node, TermDomain::Expression)
}

/// Full digest of `node` interned in `domain`.
pub fn compute_term_digest_in_domain(
    node: &TermNode,
    domain: TermDomain,
) -> Result<[u8; 32], DagError> {
    compute_term_digest_in_domain_with_limits(node, domain, DagLimits::default())
}

/// Full digest under caller-provided admission limits in the default
/// expression universe.
pub fn compute_term_digest_with_limits(
    node: &TermNode,
    limits: DagLimits,
) -> Result<[u8; 32], DagError> {
    compute_term_digest_in_domain_with_limits(node, TermDomain::Expression, limits)
}

/// Full digest under caller-provided admission limits and declared domain.
pub fn compute_term_digest_in_domain_with_limits(
    node: &TermNode,
    domain: TermDomain,
    limits: DagLimits,
) -> Result<[u8; 32], DagError> {
    validate_node_limits(node, limits)?;
    let (digest, _) = compute_term_identity_unchecked(node, domain)?;
    Ok(digest)
}

fn compute_term_identity_unchecked(
    node: &TermNode,
    domain: TermDomain,
) -> Result<([u8; 32], TermId), DagError> {
    let digest = *hash_term_preimage(node, domain)?.finalize().as_bytes();
    let mut raw_bytes = [0u8; 8];
    raw_bytes.copy_from_slice(&digest[..8]);
    let term_id = TermId::new(u64::from_le_bytes(raw_bytes)).map_err(|_| DagError::ZeroDigest)?;
    Ok((digest, term_id))
}

fn compute_term_id_unchecked(node: &TermNode, domain: TermDomain) -> Result<TermId, DagError> {
    let (_, term_id) = compute_term_identity_unchecked(node, domain)?;
    Ok(term_id)
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

fn join_numeric_sorts(lhs: &Sort, rhs: &Sort) -> Option<Sort> {
    fn rank(sort: &Sort) -> Option<u8> {
        match sort {
            Sort::Integer => Some(0),
            Sort::Rational => Some(1),
            Sort::Real => Some(2),
            Sort::Complex => Some(3),
            Sort::Scalar => Some(4),
            Sort::Boolean
            | Sort::Matrix { .. }
            | Sort::Set
            | Sort::Function { .. }
            | Sort::Unknown => None,
        }
    }

    match rank(lhs)?.max(rank(rhs)?) {
        0 => Some(Sort::Integer),
        1 => Some(Sort::Rational),
        2 => Some(Sort::Real),
        3 => Some(Sort::Complex),
        4 => Some(Sort::Scalar),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactRealSign {
    Negative,
    Zero,
    Positive,
}

fn rational_exponent_is_non_positive_integer(node: &TermNode) -> bool {
    match node {
        TermNode::Integer(value) => value.is_zero() || value.is_negative(),
        TermNode::Rational(value) => {
            value.is_integer() && (value.numer().is_zero() || value.numer().is_negative())
        }
        _ => false,
    }
}

fn exact_real_sign(node: &TermNode) -> Option<ExactRealSign> {
    let integer_sign = |value: &BigInt| {
        if value.is_negative() {
            ExactRealSign::Negative
        } else if value.is_zero() {
            ExactRealSign::Zero
        } else {
            ExactRealSign::Positive
        }
    };

    match node {
        TermNode::Integer(value) => Some(integer_sign(value)),
        TermNode::Rational(value) => Some(integer_sign(value.numer())),
        TermNode::Const(Constant::Pi | Constant::E | Constant::Infinity) => {
            Some(ExactRealSign::Positive)
        }
        TermNode::Const(Constant::NegativeInfinity) => Some(ExactRealSign::Negative),
        TermNode::Sym(_)
        | TermNode::Const(Constant::I | Constant::ComplexInfinity | Constant::NaN)
        | TermNode::Add(_)
        | TermNode::Mul(_)
        | TermNode::Pow(..)
        | TermNode::Function(..)
        | TermNode::Lambda(..) => None,
    }
}

fn infer_power_sort(
    base_sort: &Sort,
    exponent_sort: &Sort,
    base_node: &TermNode,
    exponent_node: &TermNode,
) -> Result<Sort, DagError> {
    if !base_sort.is_numeric() {
        return Err(DagError::SortMismatch {
            operation: "power base",
            actual: base_sort.clone(),
        });
    }
    if !exponent_sort.is_numeric() {
        return Err(DagError::SortMismatch {
            operation: "power exponent",
            actual: exponent_sort.clone(),
        });
    }

    if exponent_sort == &Sort::Integer {
        return Ok(match base_sort {
            Sort::Integer if matches!(exponent_node, TermNode::Integer(value) if !value.is_negative()) => {
                Sort::Integer
            }
            Sort::Integer => Sort::Rational,
            Sort::Rational => Sort::Rational,
            Sort::Real => Sort::Real,
            Sort::Complex => Sort::Complex,
            Sort::Scalar => Sort::Scalar,
            Sort::Boolean
            | Sort::Matrix { .. }
            | Sort::Set
            | Sort::Function { .. }
            | Sort::Unknown => Sort::Scalar,
        });
    }

    if matches!(exponent_sort, Sort::Rational | Sort::Real) {
        return Ok(match base_sort {
            Sort::Integer | Sort::Rational | Sort::Real => match exact_real_sign(base_node) {
                Some(ExactRealSign::Positive) => Sort::Real,
                Some(ExactRealSign::Negative) | None => Sort::Complex,
                // Zero raised to an arbitrary exact real can be undefined when
                // the exponent is non-positive, so no narrower sort is justified.
                Some(ExactRealSign::Zero) => Sort::Scalar,
            },
            Sort::Complex => Sort::Complex,
            Sort::Scalar => Sort::Scalar,
            Sort::Boolean
            | Sort::Matrix { .. }
            | Sort::Set
            | Sort::Function { .. }
            | Sort::Unknown => Sort::Scalar,
        });
    }

    Ok(match (base_sort, exponent_sort) {
        (Sort::Scalar, _) | (_, Sort::Scalar) => Sort::Scalar,
        (_, Sort::Complex) => Sort::Complex,
        _ => Sort::Scalar,
    })
}

struct InternRequest {
    term_id: TermId,
    digest: [u8; 32],
    domain: TermDomain,
    node_depth: usize,
    payload_bytes: usize,
    limits: DagLimits,
}

struct SortInferenceState {
    visiting: HashSet<TermId>,
    memo: HashMap<TermId, Sort>,
    traversed: usize,
    traversed_payload_bytes: usize,
    limits: DagLimits,
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

fn validate_lambda_surface(args: &[Expr]) -> Result<(), DagError> {
    const NAME: &str = "Lambda";
    if args.len() < 2 {
        return Err(DagError::MalformedBinder {
            name: NAME,
            reason: "expected parameters followed by a body",
        });
    }
    let parameters = &args[..args.len() - 1];
    if parameters
        .iter()
        .all(|parameter| matches!(parameter, Expr::Sym(_)))
    {
        unique_binder_parameter_names(parameters)?;
        return Ok(());
    }
    if args.len() == 2
        && let Expr::Function(name, tuple_args) = &args[0]
        && name == "Tuple"
    {
        if tuple_args.is_empty() {
            return Err(DagError::MalformedBinder {
                name: NAME,
                reason: "parameter tuple must be non-empty",
            });
        }
        if tuple_args
            .iter()
            .all(|parameter| matches!(parameter, Expr::Sym(_)))
        {
            unique_binder_parameter_names(tuple_args)?;
            return Ok(());
        }
        return Err(DagError::MalformedBinder {
            name: NAME,
            reason: "parameter tuple entries must be symbols",
        });
    }
    Err(DagError::MalformedBinder {
        name: NAME,
        reason: "parameters must be symbols or a tuple of symbols",
    })
}

fn unique_binder_parameter_names(parameters: &[Expr]) -> Result<(), DagError> {
    let mut seen = HashSet::new();
    seen.try_reserve(parameters.len())
        .map_err(|_| DagError::AllocationFailure)?;
    for parameter in parameters {
        if let Expr::Sym(symbol) = parameter {
            if !seen.insert(symbol.name.as_str()) {
                return Err(DagError::MalformedBinder {
                    name: "Lambda",
                    reason: "parameter names must be unique",
                });
            }
        }
    }
    Ok(())
}

/// Binder parameters from well-formed Lambda surface.
///
/// Multi-argument `Lambda(x, y, body)` and `Lambda(Tuple(x, y), body)` both
/// yield `[x, y]`. Returns `None` for surface that has not passed
/// [`validate_lambda_surface`].
fn lambda_symbol_parameters(args: &[Expr]) -> Result<Option<Vec<Symbol>>, DagError> {
    if args.len() < 2 {
        return Ok(None);
    }
    if args.len() == 2
        && let Expr::Function(name, tuple_args) = &args[0]
        && name == "Tuple"
    {
        let mut parameters = Vec::new();
        parameters
            .try_reserve(tuple_args.len())
            .map_err(|_| DagError::AllocationFailure)?;
        for parameter in tuple_args {
            match parameter {
                Expr::Sym(symbol) => parameters.push(symbol.clone()),
                _ => return Ok(None),
            }
        }
        return Ok(Some(parameters));
    }
    let mut parameters = Vec::new();
    parameters
        .try_reserve(args.len() - 1)
        .map_err(|_| DagError::AllocationFailure)?;
    for parameter in &args[..args.len() - 1] {
        match parameter {
            Expr::Sym(symbol) => parameters.push(symbol.clone()),
            _ => return Ok(None),
        }
    }
    Ok(Some(parameters))
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
/// Uses the default [`TermDomain::Expression`] intern universe.
pub fn compute_term_id(node: &TermNode) -> Result<TermId, DagError> {
    compute_term_id_in_domain(node, TermDomain::Expression)
}

/// Computes a stable content ID interned in `domain`.
pub fn compute_term_id_in_domain(node: &TermNode, domain: TermDomain) -> Result<TermId, DagError> {
    compute_term_id_in_domain_with_limits(node, domain, DagLimits::default())
}

/// Computes a stable content ID under caller-provided admission limits.
pub fn compute_term_id_with_limits(node: &TermNode, limits: DagLimits) -> Result<TermId, DagError> {
    compute_term_id_in_domain_with_limits(node, TermDomain::Expression, limits)
}

/// Computes a stable content ID under caller-provided limits and domain.
pub fn compute_term_id_in_domain_with_limits(
    node: &TermNode,
    domain: TermDomain,
    limits: DagLimits,
) -> Result<TermId, DagError> {
    validate_node_limits(node, limits)?;
    compute_term_id_unchecked(node, domain)
}

/// An arena-interned Semantic Term DAG with stable identity and acyclicity invariants.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermDag {
    nodes: HashMap<TermId, TermNode>,
    depths: HashMap<TermId, usize>,
    digests: HashMap<TermId, [u8; 32]>,
    domains: HashMap<TermId, TermDomain>,
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

    /// Full 32-byte preimage digest stored beside the truncated [`TermId`].
    pub fn term_digest(&self, id: TermId) -> Option<[u8; 32]> {
        self.digests.get(&id).copied()
    }

    /// Declared intern domain stored beside the truncated [`TermId`].
    pub fn term_domain(&self, id: TermId) -> Option<TermDomain> {
        self.domains.get(&id).copied()
    }

    /// Interns a [`TermNode`] in the default [`TermDomain::Expression`] universe.
    /// Fails closed if any child [`TermId`] is not already present in the DAG.
    pub fn insert_node(&mut self, node: TermNode) -> Result<TermId, DagError> {
        self.insert_node_in_domain(node, TermDomain::Expression)
    }

    /// Interns one node under caller-provided admission limits.
    pub fn insert_node_with_limits(
        &mut self,
        node: TermNode,
        limits: DagLimits,
    ) -> Result<TermId, DagError> {
        self.insert_node_in_domain_with_limits(node, TermDomain::Expression, limits)
    }

    /// Interns `node` in the declared intern universe `domain`.
    pub fn insert_node_in_domain(
        &mut self,
        node: TermNode,
        domain: TermDomain,
    ) -> Result<TermId, DagError> {
        self.insert_node_in_domain_with_limits(node, domain, DagLimits::default())
    }

    /// Interns `node` in `domain` under caller-provided admission limits.
    pub fn insert_node_in_domain_with_limits(
        &mut self,
        node: TermNode,
        domain: TermDomain,
        limits: DagLimits,
    ) -> Result<TermId, DagError> {
        validate_node_limits(&node, limits)?;
        self.validate_child_links(&node)?;
        self.validate_declared_domain(&node, domain)?;
        let node_depth = self.prospective_node_depth(&node, limits)?;
        let (digest, term_id) = compute_term_identity_unchecked(&node, domain)?;
        let payload_bytes = node_payload_bytes(&node)?;
        self.intern_prehashed_node(
            node,
            InternRequest {
                term_id,
                digest,
                domain,
                node_depth,
                payload_bytes,
                limits,
            },
        )
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

    fn validate_declared_domain(
        &self,
        node: &TermNode,
        domain: TermDomain,
    ) -> Result<(), DagError> {
        match domain {
            TermDomain::Expression => Ok(()),
            TermDomain::Integer => match node {
                TermNode::Integer(_) | TermNode::Sym(_) => Ok(()),
                TermNode::Rational(value) if value.is_integer() => Ok(()),
                TermNode::Pow(base, exponent) => {
                    let base_node = self.get(*base).ok_or(DagError::UnknownId(*base))?;
                    let exponent_node =
                        self.get(*exponent).ok_or(DagError::UnknownId(*exponent))?;
                    if matches!(exact_real_sign(base_node), Some(ExactRealSign::Zero))
                        && matches!(exact_real_sign(exponent_node), Some(ExactRealSign::Zero))
                    {
                        return Err(DagError::DomainIncompatible {
                            domain,
                            reason: "zero to the power of zero is not an Integer inhabitant",
                        });
                    }
                    match exponent_node {
                        TermNode::Integer(value) if value.is_negative() => {
                            Err(DagError::DomainIncompatible {
                                domain,
                                reason: "negative integer exponent is not closed in Integer",
                            })
                        }
                        TermNode::Rational(value)
                            if !value.is_integer() || value.numer().is_negative() =>
                        {
                            Err(DagError::DomainIncompatible {
                                domain,
                                reason: "non-natural rational exponent is not closed in Integer",
                            })
                        }
                        TermNode::Const(_) => Err(DagError::DomainIncompatible {
                            domain,
                            reason: "constant exponent is not closed in Integer",
                        }),
                        TermNode::Function(..) | TermNode::Lambda(..) => {
                            Err(DagError::DomainIncompatible {
                                domain,
                                reason: "function or binder exponent is not closed in Integer",
                            })
                        }
                        _ => Ok(()),
                    }
                }
                TermNode::Add(_) | TermNode::Mul(_) => Ok(()),
                TermNode::Rational(_)
                | TermNode::Const(_)
                | TermNode::Function(..)
                | TermNode::Lambda(..) => Err(DagError::DomainIncompatible {
                    domain,
                    reason: "operator payload is not an Integer inhabitant",
                }),
            },
            TermDomain::Rational => match node {
                TermNode::Integer(_) | TermNode::Rational(_) | TermNode::Sym(_) => Ok(()),
                TermNode::Pow(base, exponent) => {
                    let base_node = self.get(*base).ok_or(DagError::UnknownId(*base))?;
                    let exponent_node =
                        self.get(*exponent).ok_or(DagError::UnknownId(*exponent))?;
                    if matches!(exact_real_sign(base_node), Some(ExactRealSign::Zero))
                        && rational_exponent_is_non_positive_integer(exponent_node)
                    {
                        return Err(DagError::DomainIncompatible {
                            domain,
                            reason: "zero to a non-positive integer power is not a Rational inhabitant",
                        });
                    }
                    match exponent_node {
                        TermNode::Rational(value) if !value.is_integer() => {
                            Err(DagError::DomainIncompatible {
                                domain,
                                reason: "non-integer rational exponent is not closed in Rational",
                            })
                        }
                        TermNode::Const(_) => Err(DagError::DomainIncompatible {
                            domain,
                            reason: "constant exponent is not closed in Rational",
                        }),
                        TermNode::Function(..) | TermNode::Lambda(..) => {
                            Err(DagError::DomainIncompatible {
                                domain,
                                reason: "function or binder exponent is not closed in Rational",
                            })
                        }
                        _ => Ok(()),
                    }
                }
                TermNode::Add(_) | TermNode::Mul(_) => Ok(()),
                TermNode::Const(_) | TermNode::Function(..) | TermNode::Lambda(..) => {
                    Err(DagError::DomainIncompatible {
                        domain,
                        reason: "operator payload is not a Rational inhabitant",
                    })
                }
            },
            TermDomain::Real => match node {
                TermNode::Const(Constant::I | Constant::ComplexInfinity) => {
                    Err(DagError::DomainIncompatible {
                        domain,
                        reason: "complex constant is not a Real inhabitant",
                    })
                }
                TermNode::Lambda(..) => Err(DagError::DomainIncompatible {
                    domain,
                    reason: "binder is not a Real inhabitant",
                }),
                TermNode::Const(_)
                | TermNode::Integer(_)
                | TermNode::Rational(_)
                | TermNode::Sym(_)
                | TermNode::Add(_)
                | TermNode::Mul(_)
                | TermNode::Pow(..)
                | TermNode::Function(..) => Ok(()),
            },
            TermDomain::Complex => Ok(()),
        }
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
        request: InternRequest,
    ) -> Result<TermId, DagError> {
        let InternRequest {
            term_id,
            digest,
            domain,
            node_depth,
            payload_bytes,
            limits,
        } = request;
        if let Some(existing_digest) = self.digests.get(&term_id) {
            let existing = self
                .nodes
                .get(&term_id)
                .ok_or(DagError::UnknownId(term_id))?;
            let existing_domain = self
                .domains
                .get(&term_id)
                .copied()
                .ok_or(DagError::UnknownId(term_id))?;
            if existing_digest != &digest || existing != &node || existing_domain != domain {
                return Err(DagError::HashCollision(term_id));
            }
        } else {
            if self.nodes.contains_key(&term_id) {
                return Err(DagError::HashCollision(term_id));
            }
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
            self.digests
                .try_reserve(1)
                .map_err(|_| DagError::AllocationFailure)?;
            self.domains
                .try_reserve(1)
                .map_err(|_| DagError::AllocationFailure)?;
            self.nodes.insert(term_id, node);
            self.depths.insert(term_id, node_depth);
            self.digests.insert(term_id, digest);
            self.domains.insert(term_id, domain);
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
                    self.digests.remove(&id).ok_or(DagError::UnknownId(id))?;
                    self.domains.remove(&id).ok_or(DagError::UnknownId(id))?;
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
                if name == "Lambda" {
                    validate_lambda_surface(args)?;
                    let parameters =
                        lambda_symbol_parameters(args)?.ok_or(DagError::MalformedBinder {
                            name: "Lambda",
                            reason: "parameters must be symbols or a tuple of symbols",
                        })?;
                    let body = args.last().ok_or(DagError::MalformedBinder {
                        name: "Lambda",
                        reason: "expected parameters followed by a body",
                    })?;
                    let child_depth = next_depth(depth, limits.max_depth)?;
                    let body_id =
                        self.insert_expr_internal(body, child_depth, limits, traversed, inserted)?;
                    return self.insert_node_tracking(
                        TermNode::Lambda(parameters, body_id),
                        limits,
                        inserted,
                    );
                }
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
        self.validate_declared_domain(&node, TermDomain::Expression)?;
        let node_depth = self.prospective_node_depth(&node, limits)?;
        let (digest, expected_id) = compute_term_identity_unchecked(&node, TermDomain::Expression)?;
        let payload_bytes = node_payload_bytes(&node)?;
        let is_new = !self.nodes.contains_key(&expected_id);
        if is_new {
            inserted
                .try_reserve(1)
                .map_err(|_| DagError::AllocationFailure)?;
        }
        let actual_id = self.intern_prehashed_node(
            node,
            InternRequest {
                term_id: expected_id,
                digest,
                domain: TermDomain::Expression,
                node_depth,
                payload_bytes,
                limits,
            },
        )?;
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

    /// Infers a conservative mathematical sort for the interned term under default limits.
    pub fn infer_sort(&self, id: TermId) -> Result<Sort, DagError> {
        self.infer_sort_with_limits(id, DagLimits::default())
    }

    /// Infers a conservative mathematical sort under caller-provided limits.
    pub fn infer_sort_with_limits(&self, id: TermId, limits: DagLimits) -> Result<Sort, DagError> {
        let mut state = SortInferenceState {
            visiting: HashSet::new(),
            memo: HashMap::new(),
            traversed: 0,
            traversed_payload_bytes: 0,
            limits,
        };
        self.infer_sort_internal(id, &mut state, 0)
    }

    fn infer_sort_internal(
        &self,
        id: TermId,
        state: &mut SortInferenceState,
        current_depth: usize,
    ) -> Result<Sort, DagError> {
        let limits = state.limits;
        if current_depth > limits.max_depth {
            return Err(DagError::DepthExceeded(limits.max_depth));
        }
        let subtree_depth = self
            .depths
            .get(&id)
            .copied()
            .ok_or(DagError::UnknownId(id))?;
        let deepest_level = current_depth
            .checked_add(subtree_depth.saturating_sub(1))
            .ok_or(DagError::DepthExceeded(limits.max_depth))?;
        if deepest_level > limits.max_depth {
            return Err(DagError::DepthExceeded(limits.max_depth));
        }
        if let Some(sort) = state.memo.get(&id) {
            return Ok(sort.clone());
        }
        if state.traversed >= limits.max_traversal_nodes {
            return Err(DagError::TraversalLimitExceeded(limits.max_traversal_nodes));
        }
        state.traversed += 1;
        if state.visiting.contains(&id) {
            return Err(DagError::CycleDetected(id));
        }
        state
            .visiting
            .try_reserve(1)
            .map_err(|_| DagError::AllocationFailure)?;
        state.visiting.insert(id);

        let node = self.get(id).ok_or(DagError::UnknownId(id))?;
        validate_node_limits(node, limits)?;
        charge_total_payload(
            &mut state.traversed_payload_bytes,
            node_payload_bytes(node)?,
            limits.max_total_payload_bytes,
        )?;

        let computed_sort = match node {
            TermNode::Integer(_) => Sort::Integer,
            TermNode::Rational(_) => Sort::Rational,
            TermNode::Const(c) => match c {
                Constant::Pi | Constant::E | Constant::Infinity | Constant::NegativeInfinity => {
                    Sort::Real
                }
                Constant::I | Constant::ComplexInfinity => Sort::Complex,
                Constant::NaN => Sort::Scalar,
            },
            TermNode::Sym(_) => Sort::Scalar,
            TermNode::Add(ids) | TermNode::Mul(ids) => {
                let child_depth = if ids.is_empty() {
                    current_depth
                } else {
                    next_depth(current_depth, limits.max_depth)?
                };
                let operation = if matches!(node, TermNode::Add(_)) {
                    "addition"
                } else {
                    "multiplication"
                };
                let mut current = Sort::Integer;
                for &child_id in ids {
                    let child_sort = self.infer_sort_internal(child_id, state, child_depth)?;
                    current = join_numeric_sorts(&current, &child_sort).ok_or(
                        DagError::SortMismatch {
                            operation,
                            actual: child_sort,
                        },
                    )?;
                }
                current
            }
            TermNode::Pow(base, exponent) => {
                let child_depth = next_depth(current_depth, limits.max_depth)?;
                let base_sort = self.infer_sort_internal(*base, state, child_depth)?;
                let exp_sort = self.infer_sort_internal(*exponent, state, child_depth)?;
                let base_node = self.get(*base).ok_or(DagError::UnknownId(*base))?;
                let exponent_node = self.get(*exponent).ok_or(DagError::UnknownId(*exponent))?;
                infer_power_sort(&base_sort, &exp_sort, base_node, exponent_node)?
            }
            TermNode::Function(_, ids) => {
                let child_depth = if ids.is_empty() {
                    current_depth
                } else {
                    next_depth(current_depth, limits.max_depth)?
                };
                for &child_id in ids {
                    self.infer_sort_internal(child_id, state, child_depth)?;
                }
                Sort::Scalar
            }
            TermNode::Lambda(params, body) => {
                let child_depth = next_depth(current_depth, limits.max_depth)?;
                let body_sort = self.infer_sort_internal(*body, state, child_depth)?;
                let mut dom = Vec::new();
                dom.try_reserve(params.len())
                    .map_err(|_| DagError::AllocationFailure)?;
                dom.resize(params.len(), Sort::Scalar);
                Sort::Function {
                    dom,
                    codom: Box::new(body_sort),
                }
            }
        };

        state.visiting.remove(&id);
        state
            .memo
            .try_reserve(1)
            .map_err(|_| DagError::AllocationFailure)?;
        state.memo.insert(id, computed_sort.clone());
        Ok(computed_sort)
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
    fn interned_term_id_is_confirmed_by_full_preimage_digest() {
        let mut dag = TermDag::new();
        let x = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let y = dag.insert_node(TermNode::Sym(Symbol::new("y"))).unwrap();
        let x_again = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();

        let digest_x = dag.term_digest(x).expect("interned terms store a digest");
        let digest_y = dag.term_digest(y).expect("interned terms store a digest");
        assert_eq!(x, x_again);
        assert_eq!(digest_x, dag.term_digest(x_again).unwrap());
        assert_ne!(digest_x, digest_y);
        assert_eq!(
            x.raw(),
            u64::from_le_bytes(digest_x[..8].try_into().unwrap())
        );
        assert_eq!(
            digest_x,
            compute_term_digest(&TermNode::Sym(Symbol::new("x"))).unwrap()
        );

        let colliding = TermNode::Sym(Symbol::new("y"));
        let mut forged = digest_x;
        forged[8] ^= 1;
        let payload = node_payload_bytes(&colliding).unwrap();
        assert_eq!(
            dag.intern_prehashed_node(
                colliding,
                InternRequest {
                    term_id: x,
                    digest: forged,
                    domain: TermDomain::Expression,
                    node_depth: 1,
                    payload_bytes: payload,
                    limits: DagLimits::default(),
                },
            ),
            Err(DagError::HashCollision(x))
        );
        assert_eq!(dag.term_digest(x), Some(digest_x));
        assert_eq!(dag.term_domain(x), Some(TermDomain::Expression));
        assert_eq!(dag.get(x), Some(&TermNode::Sym(Symbol::new("x"))));
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
    fn well_formed_symbol_lambda_lowers_to_binder_node() {
        let expression = Expr::Function(
            "Lambda".to_string(),
            vec![Expr::symbol("x"), Expr::symbol("x")],
        );
        let mut dag = TermDag::new();
        let root = dag.insert_expr(&expression).unwrap();
        assert!(matches!(
            dag.get(root),
            Some(TermNode::Lambda(parameters, _)) if parameters == &vec![Symbol::new("x")]
        ));
        assert_eq!(dag.to_expr(root).unwrap(), expression);
        assert_eq!(
            dag.infer_sort(root).unwrap(),
            Sort::Function {
                dom: vec![Sort::Scalar],
                codom: Box::new(Sort::Scalar),
            }
        );

        let mut direct = TermDag::new();
        let body = direct.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let direct_root = direct
            .insert_node(TermNode::Lambda(vec![Symbol::new("x")], body))
            .unwrap();
        assert_eq!(root, direct_root);

        let two_param = Expr::Function(
            "Lambda".to_string(),
            vec![Expr::symbol("x"), Expr::symbol("y"), Expr::symbol("x")],
        );
        let two_root = dag.insert_expr(&two_param).unwrap();
        assert!(matches!(
            dag.get(two_root),
            Some(TermNode::Lambda(parameters, _))
                if parameters == &vec![Symbol::new("x"), Symbol::new("y")]
        ));
        assert_eq!(dag.to_expr(two_root).unwrap(), two_param);
    }

    #[test]
    fn tuple_parameter_lambda_interns_as_the_same_binder_as_multi_arg() {
        let tuple_lambda = Expr::Function(
            "Lambda".to_string(),
            vec![
                Expr::Function(
                    "Tuple".to_string(),
                    vec![Expr::symbol("x"), Expr::symbol("y")],
                ),
                Expr::symbol("x"),
            ],
        );
        let multi_arg = Expr::Function(
            "Lambda".to_string(),
            vec![Expr::symbol("x"), Expr::symbol("y"), Expr::symbol("x")],
        );
        let mut dag = TermDag::new();
        let tuple_root = dag.insert_expr(&tuple_lambda).unwrap();
        let multi_root = dag.insert_expr(&multi_arg).unwrap();
        assert_eq!(tuple_root, multi_root);
        assert!(matches!(
            dag.get(tuple_root),
            Some(TermNode::Lambda(parameters, _))
                if parameters == &vec![Symbol::new("x"), Symbol::new("y")]
        ));
        // Lifting uses the multi-argument surface, not the Tuple spelling.
        assert_eq!(dag.to_expr(tuple_root).unwrap(), multi_arg);
        assert_ne!(dag.to_expr(tuple_root).unwrap(), tuple_lambda);
    }

    #[test]
    fn malformed_lambda_surface_is_refused_instead_of_interned_as_function() {
        let mut dag = TermDag::new();
        let existing = dag.insert_node(TermNode::Sym(Symbol::new("keep"))).unwrap();
        let before = dag.len();
        let malformed = Expr::Function(
            "Lambda".to_string(),
            vec![Expr::Integer(BigInt::from(0)), Expr::symbol("x")],
        );
        assert_eq!(
            dag.insert_expr(&malformed),
            Err(DagError::MalformedBinder {
                name: "Lambda",
                reason: "parameters must be symbols or a tuple of symbols",
            })
        );
        assert_eq!(dag.len(), before);
        assert!(dag.get(existing).is_some());

        assert_eq!(
            dag.insert_expr(&Expr::Function(
                "Lambda".to_string(),
                vec![Expr::symbol("x")]
            )),
            Err(DagError::MalformedBinder {
                name: "Lambda",
                reason: "expected parameters followed by a body",
            })
        );
        assert_eq!(
            dag.insert_expr(&Expr::Function(
                "Lambda".to_string(),
                vec![
                    Expr::Function(
                        "Tuple".to_string(),
                        vec![Expr::Integer(BigInt::from(1)), Expr::symbol("x")]
                    ),
                    Expr::symbol("x"),
                ]
            )),
            Err(DagError::MalformedBinder {
                name: "Lambda",
                reason: "parameter tuple entries must be symbols",
            })
        );
        assert_eq!(dag.len(), before);
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
    fn memoized_sort_inference_still_enforces_the_deeper_parent_path() {
        let mut dag = TermDag::new();
        let leaf = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let tail_1 = dag.insert_node(TermNode::Add(vec![leaf])).unwrap();
        let tail_2 = dag.insert_node(TermNode::Add(vec![tail_1])).unwrap();
        let deep_1 = dag.insert_node(TermNode::Add(vec![tail_2])).unwrap();
        let deep_2 = dag.insert_node(TermNode::Add(vec![deep_1])).unwrap();
        let deep_3 = dag.insert_node(TermNode::Add(vec![deep_2])).unwrap();
        let root = dag
            .insert_node(TermNode::Add(vec![tail_2, deep_3]))
            .unwrap();
        let limits = DagLimits {
            max_depth: 5,
            ..DagLimits::default()
        };

        assert_eq!(
            dag.infer_sort_with_limits(root, limits),
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

    #[test]
    fn sort_inference_covers_numeric_tower_and_integer_powers() {
        let mut dag = TermDag::new();
        let i1 = dag
            .insert_node(TermNode::Integer(BigInt::from(42)))
            .unwrap();
        let two = dag.insert_node(TermNode::Integer(BigInt::from(2))).unwrap();
        let i2 = dag
            .insert_node(TermNode::Integer(BigInt::from(-3)))
            .unwrap();
        let q1 = dag
            .insert_node(TermNode::Rational(BigRational::new(
                BigInt::from(1),
                BigInt::from(2),
            )))
            .unwrap();
        let pi = dag.insert_node(TermNode::Const(Constant::Pi)).unwrap();
        let img = dag.insert_node(TermNode::Const(Constant::I)).unwrap();

        assert_eq!(dag.infer_sort(i1).unwrap(), crate::sort::Sort::Integer);
        assert_eq!(dag.infer_sort(q1).unwrap(), crate::sort::Sort::Rational);
        assert_eq!(dag.infer_sort(pi).unwrap(), crate::sort::Sort::Real);
        assert_eq!(dag.infer_sort(img).unwrap(), crate::sort::Sort::Complex);

        let add_int = dag.insert_node(TermNode::Add(vec![i1, i2])).unwrap();
        assert_eq!(dag.infer_sort(add_int).unwrap(), crate::sort::Sort::Integer);

        let add_rat = dag.insert_node(TermNode::Add(vec![i1, q1])).unwrap();
        assert_eq!(
            dag.infer_sort(add_rat).unwrap(),
            crate::sort::Sort::Rational
        );

        let add_real = dag.insert_node(TermNode::Add(vec![q1, pi])).unwrap();
        assert_eq!(dag.infer_sort(add_real).unwrap(), crate::sort::Sort::Real);

        let add_cplx = dag.insert_node(TermNode::Add(vec![pi, img])).unwrap();
        assert_eq!(
            dag.infer_sort(add_cplx).unwrap(),
            crate::sort::Sort::Complex
        );

        // Pow: 42^2 is Integer, 42^(-3) is Rational
        let pow_pos = dag.insert_node(TermNode::Pow(i1, two)).unwrap();
        assert_eq!(dag.infer_sort(pow_pos).unwrap(), crate::sort::Sort::Integer);

        let pow_neg = dag.insert_node(TermNode::Pow(i1, i2)).unwrap();
        assert_eq!(
            dag.infer_sort(pow_neg).unwrap(),
            crate::sort::Sort::Rational
        );
    }

    #[test]
    fn sort_inference_preserves_scalar_and_rejects_nonnumeric_operands() {
        let mut dag = TermDag::new();
        let x = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let one = dag.insert_node(TermNode::Integer(BigInt::from(1))).unwrap();

        let sum = dag.insert_node(TermNode::Add(vec![x, one])).unwrap();
        let product = dag.insert_node(TermNode::Mul(vec![x, one])).unwrap();
        assert_eq!(dag.infer_sort(sum).unwrap(), Sort::Scalar);
        assert_eq!(dag.infer_sort(product).unwrap(), Sort::Scalar);

        let lambda = dag
            .insert_node(TermNode::Lambda(vec![Symbol::new("p")], one))
            .unwrap();
        let invalid = dag.insert_node(TermNode::Add(vec![lambda, one])).unwrap();
        assert_eq!(
            dag.infer_sort(invalid),
            Err(DagError::SortMismatch {
                operation: "addition",
                actual: Sort::Function {
                    dom: vec![Sort::Scalar],
                    codom: Box::new(Sort::Integer),
                },
            })
        );
    }

    #[test]
    fn power_sort_inference_respects_exact_sign_and_branch_uncertainty() {
        let mut dag = TermDag::new();
        let negative_half = dag
            .insert_node(TermNode::Rational(BigRational::new(
                BigInt::from(-1),
                BigInt::from(2),
            )))
            .unwrap();
        let positive_half = dag
            .insert_node(TermNode::Rational(BigRational::new(
                BigInt::from(1),
                BigInt::from(2),
            )))
            .unwrap();
        let square_root_of_negative = dag
            .insert_node(TermNode::Pow(negative_half, positive_half))
            .unwrap();
        let square_root_of_positive = dag
            .insert_node(TermNode::Pow(positive_half, positive_half))
            .unwrap();

        assert_eq!(
            dag.infer_sort(square_root_of_negative).unwrap(),
            Sort::Complex
        );
        assert_eq!(dag.infer_sort(square_root_of_positive).unwrap(), Sort::Real);

        let minus_three = dag
            .insert_node(TermNode::Integer(BigInt::from(-3)))
            .unwrap();
        let rational_integer_power = dag
            .insert_node(TermNode::Pow(positive_half, minus_three))
            .unwrap();
        assert_eq!(
            dag.infer_sort(rational_integer_power).unwrap(),
            Sort::Rational
        );

        let pi = dag.insert_node(TermNode::Const(Constant::Pi)).unwrap();
        let unknown_real_sign = dag
            .insert_node(TermNode::Add(vec![pi, minus_three]))
            .unwrap();
        let branch_sensitive = dag
            .insert_node(TermNode::Pow(unknown_real_sign, positive_half))
            .unwrap();
        assert_eq!(dag.infer_sort(branch_sensitive).unwrap(), Sort::Complex);
    }

    #[test]
    fn sort_inference_enforces_local_and_aggregate_limits() {
        let mut dag = TermDag::new();
        let integer = dag.insert_node(TermNode::Integer(BigInt::from(2))).unwrap();
        let add = dag.insert_node(TermNode::Add(vec![integer])).unwrap();
        let symbol = dag.insert_node(TermNode::Sym(Symbol::new("xy"))).unwrap();
        let lambda = dag
            .insert_node(TermNode::Lambda(
                vec![Symbol::new("a"), Symbol::new("b")],
                integer,
            ))
            .unwrap();

        assert_eq!(
            dag.infer_sort_with_limits(
                integer,
                DagLimits {
                    max_numeric_bits: 1,
                    ..DagLimits::default()
                }
            ),
            Err(DagError::NumericPayloadLimitExceeded(1))
        );
        assert_eq!(
            dag.infer_sort_with_limits(
                add,
                DagLimits {
                    max_arity: 0,
                    ..DagLimits::default()
                }
            ),
            Err(DagError::ArityLimitExceeded(0))
        );
        assert_eq!(
            dag.infer_sort_with_limits(
                lambda,
                DagLimits {
                    max_arity: 2,
                    ..DagLimits::default()
                }
            ),
            Err(DagError::ArityLimitExceeded(2))
        );
        assert_eq!(
            dag.infer_sort_with_limits(
                symbol,
                DagLimits {
                    max_payload_bytes: 1,
                    ..DagLimits::default()
                }
            ),
            Err(DagError::PayloadLimitExceeded(1))
        );

        let one_byte_symbol = dag.insert_node(TermNode::Sym(Symbol::new("z"))).unwrap();
        let wrapper = dag
            .insert_node(TermNode::Add(vec![one_byte_symbol]))
            .unwrap();
        assert_eq!(
            dag.infer_sort_with_limits(
                wrapper,
                DagLimits {
                    max_total_payload_bytes: TERM_ID_SLOT_CHARGE_BYTES,
                    ..DagLimits::default()
                }
            ),
            Err(DagError::TotalPayloadLimitExceeded(
                TERM_ID_SLOT_CHARGE_BYTES
            ))
        );
    }

    #[test]
    fn declared_domain_is_intern_identity_and_not_sort() {
        let mut dag = TermDag::new();
        let two = TermNode::Integer(BigInt::from(2));
        let expression = dag.insert_node(two.clone()).unwrap();
        let integer = dag
            .insert_node_in_domain(two.clone(), TermDomain::Integer)
            .unwrap();
        let rational = dag
            .insert_node_in_domain(two, TermDomain::Rational)
            .unwrap();

        assert_ne!(expression, integer);
        assert_ne!(integer, rational);
        assert_eq!(dag.term_domain(expression), Some(TermDomain::Expression));
        assert_eq!(dag.term_domain(integer), Some(TermDomain::Integer));
        assert_eq!(dag.term_domain(rational), Some(TermDomain::Rational));
        assert_eq!(dag.infer_sort(expression).unwrap(), Sort::Integer);
        assert_eq!(dag.infer_sort(integer).unwrap(), Sort::Integer);
        assert_eq!(
            compute_term_id_in_domain(&TermNode::Integer(BigInt::from(2)), TermDomain::Integer)
                .unwrap(),
            integer
        );
    }

    #[test]
    fn integer_domain_refuses_negative_integer_power_instead_of_widening() {
        let mut dag = TermDag::new();
        let base = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(42)), TermDomain::Integer)
            .unwrap();
        let negative = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(-3)), TermDomain::Integer)
            .unwrap();
        let two = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(2)), TermDomain::Integer)
            .unwrap();

        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(base, negative), TermDomain::Integer),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Integer,
                reason: "negative integer exponent is not closed in Integer",
            })
        );
        assert!(dag.get(base).is_some());
        assert!(dag.get(negative).is_some());

        let positive_power = dag
            .insert_node_in_domain(TermNode::Pow(base, two), TermDomain::Integer)
            .unwrap();
        assert_eq!(dag.term_domain(positive_power), Some(TermDomain::Integer));
        assert_eq!(dag.infer_sort(positive_power).unwrap(), Sort::Integer);

        // Expression-domain intern still records the widening at sort inference;
        // it does not pretend the power inhabits Integer.
        let expr_base = dag
            .insert_node(TermNode::Integer(BigInt::from(42)))
            .unwrap();
        let expr_neg = dag
            .insert_node(TermNode::Integer(BigInt::from(-3)))
            .unwrap();
        let widened = dag.insert_node(TermNode::Pow(expr_base, expr_neg)).unwrap();
        assert_eq!(dag.term_domain(widened), Some(TermDomain::Expression));
        assert_eq!(dag.infer_sort(widened).unwrap(), Sort::Rational);
    }

    #[test]
    fn integer_domain_refuses_non_integer_atoms() {
        let mut dag = TermDag::new();
        let half = TermNode::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
        assert_eq!(
            dag.insert_node_in_domain(half, TermDomain::Integer),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Integer,
                reason: "operator payload is not an Integer inhabitant",
            })
        );
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Const(Constant::I), TermDomain::Integer),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Integer,
                reason: "operator payload is not an Integer inhabitant",
            })
        );
        let as_integer = dag
            .insert_node_in_domain(
                TermNode::Rational(BigRational::from_integer(BigInt::from(7))),
                TermDomain::Integer,
            )
            .unwrap();
        assert_eq!(dag.term_domain(as_integer), Some(TermDomain::Integer));
        assert_eq!(
            dag.get(as_integer),
            Some(&TermNode::Rational(BigRational::from_integer(
                BigInt::from(7)
            )))
        );
    }

    #[test]
    fn rational_domain_refuses_non_rational_payloads_and_powers() {
        let mut dag = TermDag::new();
        // Constants are not in Rational
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Const(Constant::Pi), TermDomain::Rational),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Rational,
                reason: "operator payload is not a Rational inhabitant",
            })
        );
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Const(Constant::I), TermDomain::Rational),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Rational,
                reason: "operator payload is not a Rational inhabitant",
            })
        );

        // Power with non-integer rational exponent (e.g. x^(1/2)) is not closed in Rational
        let base = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(2)), TermDomain::Rational)
            .unwrap();
        let half = dag
            .insert_node_in_domain(
                TermNode::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))),
                TermDomain::Rational,
            )
            .unwrap();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(base, half), TermDomain::Rational),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Rational,
                reason: "non-integer rational exponent is not closed in Rational",
            })
        );

        // Integer powers (including negative integers) are closed in Rational
        let neg_two = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(-2)), TermDomain::Rational)
            .unwrap();
        let power = dag
            .insert_node_in_domain(TermNode::Pow(base, neg_two), TermDomain::Rational)
            .unwrap();
        assert_eq!(dag.term_domain(power), Some(TermDomain::Rational));
    }

    #[test]
    fn real_domain_refuses_complex_constants() {
        let mut dag = TermDag::new();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Const(Constant::I), TermDomain::Real),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Real,
                reason: "complex constant is not a Real inhabitant",
            })
        );
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Const(Constant::ComplexInfinity), TermDomain::Real),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Real,
                reason: "complex constant is not a Real inhabitant",
            })
        );

        // Real constants are admitted
        let pi = dag
            .insert_node_in_domain(TermNode::Const(Constant::Pi), TermDomain::Real)
            .unwrap();
        assert_eq!(dag.term_domain(pi), Some(TermDomain::Real));
    }

    #[test]
    fn integer_domain_refuses_zero_to_the_power_of_zero() {
        let mut dag = TermDag::new();
        let zero = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(0)), TermDomain::Integer)
            .unwrap();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(zero, zero), TermDomain::Integer),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Integer,
                reason: "zero to the power of zero is not an Integer inhabitant",
            })
        );
        assert!(dag.get(zero).is_some());
    }

    #[test]
    fn integer_domain_refuses_function_and_binder_exponents() {
        let mut dag = TermDag::new();
        let base = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(2)), TermDomain::Integer)
            .unwrap();
        let arg = dag.insert_node(TermNode::Integer(BigInt::from(1))).unwrap();
        let sine = dag
            .insert_node(TermNode::Function("sin".to_string(), vec![arg]))
            .unwrap();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(base, sine), TermDomain::Integer),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Integer,
                reason: "function or binder exponent is not closed in Integer",
            })
        );

        let body = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        let binder = dag
            .insert_node(TermNode::Lambda(vec![Symbol::new("x")], body))
            .unwrap();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(base, binder), TermDomain::Integer),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Integer,
                reason: "function or binder exponent is not closed in Integer",
            })
        );
        assert!(dag.get(base).is_some());
        assert!(dag.get(sine).is_some());
        assert!(dag.get(binder).is_some());
    }

    #[test]
    fn rational_domain_refuses_zero_to_a_negative_integer_power() {
        let mut dag = TermDag::new();
        let zero = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(0)), TermDomain::Rational)
            .unwrap();
        let negative = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(-1)), TermDomain::Rational)
            .unwrap();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(zero, negative), TermDomain::Rational),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Rational,
                reason: "zero to a non-positive integer power is not a Rational inhabitant",
            })
        );

        let two = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(2)), TermDomain::Rational)
            .unwrap();
        let admitted = dag
            .insert_node_in_domain(TermNode::Pow(two, negative), TermDomain::Rational)
            .unwrap();
        assert_eq!(dag.term_domain(admitted), Some(TermDomain::Rational));
        assert!(dag.get(zero).is_some());
    }

    #[test]
    fn rational_domain_refuses_function_exponents() {
        let mut dag = TermDag::new();
        let base = dag
            .insert_node_in_domain(TermNode::Integer(BigInt::from(3)), TermDomain::Rational)
            .unwrap();
        let arg = dag.insert_node(TermNode::Integer(BigInt::from(1))).unwrap();
        let sine = dag
            .insert_node(TermNode::Function("sin".to_string(), vec![arg]))
            .unwrap();
        assert_eq!(
            dag.insert_node_in_domain(TermNode::Pow(base, sine), TermDomain::Rational),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Rational,
                reason: "function or binder exponent is not closed in Rational",
            })
        );
    }

    #[test]
    fn real_domain_refuses_lambda_binders() {
        let mut dag = TermDag::new();
        let body = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
        assert_eq!(
            dag.insert_node_in_domain(
                TermNode::Lambda(vec![Symbol::new("x")], body),
                TermDomain::Real
            ),
            Err(DagError::DomainIncompatible {
                domain: TermDomain::Real,
                reason: "binder is not a Real inhabitant",
            })
        );
        assert!(dag.get(body).is_some());
    }

    #[test]
    fn duplicate_lambda_parameter_names_are_refused() {
        let mut dag = TermDag::new();
        let existing = dag.insert_node(TermNode::Sym(Symbol::new("keep"))).unwrap();
        let before = dag.len();
        let duplicate = Expr::Function(
            "Lambda".to_string(),
            vec![Expr::symbol("x"), Expr::symbol("x"), Expr::symbol("x")],
        );
        assert_eq!(
            dag.insert_expr(&duplicate),
            Err(DagError::MalformedBinder {
                name: "Lambda",
                reason: "parameter names must be unique",
            })
        );
        assert_eq!(dag.len(), before);
        assert!(dag.get(existing).is_some());

        let duplicate_tuple = Expr::Function(
            "Lambda".to_string(),
            vec![
                Expr::Function(
                    "Tuple".to_string(),
                    vec![Expr::symbol("y"), Expr::symbol("y")],
                ),
                Expr::symbol("y"),
            ],
        );
        assert_eq!(
            dag.insert_expr(&duplicate_tuple),
            Err(DagError::MalformedBinder {
                name: "Lambda",
                reason: "parameter names must be unique",
            })
        );
        assert_eq!(dag.len(), before);
    }
}
