//! Provisional in-memory semantic workspaces for the pre-WS14 runtime slice.
//!
//! This module provides bounded structural patching and a conservative merge primitive. It is not
//! the serializable MVCC workspace, universe manifest, or independently checkable merge-certificate
//! system described by the WS14 contract.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::{DagError, Expr, Symbol, TermDag};
use fsym_proof_kernel::DerivationTree;
use fsym_simplify::simplify;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::Arc;
use thiserror::Error;

const MAX_WORKSPACE_BRANCH_NAME_BYTES: usize = 1_024;
const MAX_WORKSPACE_BINDINGS: usize = 65_536;
const MAX_WORKSPACE_DERIVATIONS: usize = 4_096;
const MAX_MERGE_RECEIPT_CONTENT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkspaceError {
    #[error("Semantic merge conflict in an overlapping binding")]
    BindingConflict { symbol: Symbol },
    #[error("Semantic merge refused because assumption contexts differ")]
    AssumptionContextMismatch {
        target_context_digest: [u8; 32],
        incoming_context_digest: [u8; 32],
    },
    #[error("Semantic patch cannot update and remove the same binding")]
    AmbiguousBindingEdit { symbol: Symbol },
    #[error("Derivation verification failed for incoming branch")]
    DerivationVerificationFailed,
    #[error("Workspace exceeds the provisional resource limits")]
    ResourceLimitExceeded,
    #[error("Workspace allocation failed")]
    AllocationFailure,
    #[error("Workspace structural validation detected an internal invariant failure")]
    StructuralInvariantFailure,
    #[error("Merge receipt serialization failed")]
    ReceiptSerializationFailed,
}

/// A semantic patch modifying workspace bindings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspacePatch {
    pub updated_bindings: HashMap<Symbol, Expr>,
    pub removed_bindings: Vec<Symbol>,
}

/// Verification receipt for a clean semantic merge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeReceipt {
    pub source_branch: String,
    pub target_branch: String,
    /// Total number of bindings in the target after the merge.
    pub merged_bindings_count: usize,
    /// Deterministic digest of the provisional post-merge binding, assumption, and derivation state.
    ///
    /// This is a content-binding receipt field, not a WS14 `SemanticMergeCertificate` or stable
    /// workspace identity.
    pub content_digest: [u8; 32],
}

/// Isolated semantic workspace holding bindings, assumptions context, and derivations.
///
/// A workspace intentionally has no blanket serde representation: silently restoring the skipped
/// assumptions context would change its semantic universe. A future persisted snapshot must use the
/// complete, versioned universe schema from the workspace contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticWorkspace {
    pub branch_name: String,
    pub bindings: HashMap<Symbol, Expr>,
    pub assumptions: Arc<ImmutableAssumptionsSnapshot>,
    pub derivations: Vec<DerivationTree>,
}

struct BoundedDigestWriter<'a> {
    hasher: &'a mut blake3::Hasher,
    bytes_written: usize,
    max_bytes: usize,
    limit_exceeded: bool,
}

impl<'a> BoundedDigestWriter<'a> {
    fn new(hasher: &'a mut blake3::Hasher, max_bytes: usize) -> Self {
        Self {
            hasher,
            bytes_written: 0,
            max_bytes,
            limit_exceeded: false,
        }
    }
}

impl Write for BoundedDigestWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes_written.checked_add(buffer.len()) else {
            self.limit_exceeded = true;
            return Err(io::Error::other("workspace receipt size limit exceeded"));
        };
        if new_len > self.max_bytes {
            self.limit_exceeded = true;
            return Err(io::Error::other("workspace receipt size limit exceeded"));
        }
        self.hasher.update(buffer);
        self.bytes_written = new_len;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn charge_receipt_bytes(total: &mut usize, additional: usize) -> Result<(), WorkspaceError> {
    *total = total
        .checked_add(additional)
        .filter(|value| *value <= MAX_MERGE_RECEIPT_CONTENT_BYTES)
        .ok_or(WorkspaceError::ResourceLimitExceeded)?;
    Ok(())
}

fn structured_digest<T: Serialize>(
    domain: &'static [u8],
    value: &T,
    total_bytes: &mut usize,
) -> Result<[u8; 32], WorkspaceError> {
    let remaining = MAX_MERGE_RECEIPT_CONTENT_BYTES
        .checked_sub(*total_bytes)
        .ok_or(WorkspaceError::ResourceLimitExceeded)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    let mut writer = BoundedDigestWriter::new(&mut hasher, remaining);
    let serialized = serde_json::to_writer(&mut writer, value);
    let bytes_written = writer.bytes_written;
    let limit_exceeded = writer.limit_exceeded;
    if serialized.is_err() {
        return Err(if limit_exceeded {
            WorkspaceError::ResourceLimitExceeded
        } else {
            WorkspaceError::ReceiptSerializationFailed
        });
    }
    charge_receipt_bytes(total_bytes, bytes_written)?;
    Ok(*hasher.finalize().as_bytes())
}

fn map_dag_error(error: DagError) -> WorkspaceError {
    match error {
        DagError::AllocationFailure => WorkspaceError::AllocationFailure,
        DagError::HashCollision(_)
        | DagError::ZeroDigest
        | DagError::DanglingChild(_)
        | DagError::CycleDetected(_)
        | DagError::UnknownId(_)
        | DagError::SortMismatch { .. }
        | DagError::DomainIncompatible { .. }
        | DagError::MalformedBinder { .. }
        | DagError::UnboundIndex(_) => WorkspaceError::StructuralInvariantFailure,
        DagError::DepthExceeded(_)
        | DagError::PayloadLengthOverflow
        | DagError::NodeLimitExceeded(_)
        | DagError::TraversalLimitExceeded(_)
        | DagError::ExpansionLimitExceeded(_)
        | DagError::ArityLimitExceeded(_)
        | DagError::PayloadLimitExceeded(_)
        | DagError::TotalPayloadLimitExceeded(_)
        | DagError::NumericPayloadLimitExceeded(_) => WorkspaceError::ResourceLimitExceeded,
    }
}

fn validate_branch_name(branch_name: &str) -> Result<(), WorkspaceError> {
    if branch_name.is_empty() || branch_name.len() > MAX_WORKSPACE_BRANCH_NAME_BYTES {
        return Err(WorkspaceError::ResourceLimitExceeded);
    }
    Ok(())
}

fn merge_content_digest(
    target_branch: &str,
    source_branch: &str,
    assumption_digest: [u8; 32],
    bindings: &[(&Symbol, &Expr)],
    derivations: &[&DerivationTree],
) -> Result<[u8; 32], WorkspaceError> {
    let target_branch_len =
        u64::try_from(target_branch.len()).map_err(|_| WorkspaceError::ResourceLimitExceeded)?;
    let source_branch_len =
        u64::try_from(source_branch.len()).map_err(|_| WorkspaceError::ResourceLimitExceeded)?;
    let binding_count =
        u64::try_from(bindings.len()).map_err(|_| WorkspaceError::ResourceLimitExceeded)?;
    let mut total_bytes = 0_usize;
    charge_receipt_bytes(&mut total_bytes, target_branch.len())?;
    charge_receipt_bytes(&mut total_bytes, source_branch.len())?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.workspace.provisional-merge-receipt.v2\0");
    hasher.update(&target_branch_len.to_le_bytes());
    hasher.update(target_branch.as_bytes());
    hasher.update(&source_branch_len.to_le_bytes());
    hasher.update(source_branch.as_bytes());
    hasher.update(&assumption_digest);
    hasher.update(&binding_count.to_le_bytes());

    let mut term_dag = TermDag::new();
    for (symbol, expression) in bindings {
        term_dag.insert_expr(expression).map_err(map_dag_error)?;
        charge_receipt_bytes(&mut total_bytes, symbol.name.len())?;
        let symbol_len =
            u64::try_from(symbol.name.len()).map_err(|_| WorkspaceError::ResourceLimitExceeded)?;
        hasher.update(&symbol_len.to_le_bytes());
        hasher.update(symbol.name.as_bytes());
        let expression_digest = structured_digest(
            b"fsym.workspace.binding-expression.v1\0",
            expression,
            &mut total_bytes,
        )?;
        hasher.update(&expression_digest);
    }

    let mut derivation_digests = Vec::new();
    derivation_digests
        .try_reserve(derivations.len())
        .map_err(|_| WorkspaceError::AllocationFailure)?;
    for derivation in derivations {
        derivation_digests.push(structured_digest(
            b"fsym.workspace.derivation.v1\0",
            derivation,
            &mut total_bytes,
        )?);
    }
    derivation_digests.sort_unstable();
    let derivation_count = u64::try_from(derivation_digests.len())
        .map_err(|_| WorkspaceError::ResourceLimitExceeded)?;
    hasher.update(&derivation_count.to_le_bytes());
    for digest in derivation_digests {
        hasher.update(&digest);
    }

    Ok(*hasher.finalize().as_bytes())
}

impl SemanticWorkspace {
    pub fn new(branch_name: impl Into<String>) -> Self {
        Self {
            branch_name: branch_name.into(),
            bindings: HashMap::new(),
            assumptions: ImmutableAssumptionsSnapshot::empty(),
            derivations: Vec::new(),
        }
    }

    /// Sets or updates a named symbolic binding.
    pub fn bind(&mut self, symbol: Symbol, expr: Expr) {
        self.bindings.insert(symbol, expr);
    }

    /// Resolves an expression by substituting workspace bindings.
    pub fn eval(&self, expr: &Expr) -> Expr {
        let substituted = expr.subs(&self.bindings);
        simplify(&substituted)
    }

    /// Forks a child branch workspace with isolated state.
    pub fn fork(&self, branch_name: impl Into<String>) -> Self {
        Self {
            branch_name: branch_name.into(),
            bindings: self.bindings.clone(),
            assumptions: self.assumptions.clone(),
            derivations: self.derivations.clone(),
        }
    }

    /// Applies a semantic patch to the workspace.
    pub fn apply_patch(&mut self, patch: &WorkspacePatch) -> Result<(), WorkspaceError> {
        for symbol in &patch.removed_bindings {
            if patch.updated_bindings.contains_key(symbol) {
                return Err(WorkspaceError::AmbiguousBindingEdit {
                    symbol: symbol.clone(),
                });
            }
        }

        let mut removed_bindings = HashSet::new();
        removed_bindings
            .try_reserve(patch.removed_bindings.len())
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        for symbol in &patch.removed_bindings {
            if self.bindings.contains_key(symbol) {
                removed_bindings.insert(symbol);
            }
        }

        let added_bindings = patch
            .updated_bindings
            .keys()
            .filter(|symbol| !self.bindings.contains_key(*symbol))
            .count();
        let resulting_bindings = self
            .bindings
            .len()
            .checked_add(added_bindings)
            .and_then(|count| count.checked_sub(removed_bindings.len()))
            .ok_or(WorkspaceError::ResourceLimitExceeded)?;
        if resulting_bindings > MAX_WORKSPACE_BINDINGS {
            return Err(WorkspaceError::ResourceLimitExceeded);
        }

        let reserve_growth = resulting_bindings.saturating_sub(self.bindings.len());
        self.bindings
            .try_reserve(reserve_growth)
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        for sym in &patch.removed_bindings {
            self.bindings.remove(sym);
        }
        for (sym, expr) in &patch.updated_bindings {
            self.bindings.insert(sym.clone(), expr.clone());
        }
        Ok(())
    }

    /// Conservatively merges one provisional in-memory branch into `self`.
    ///
    /// Assumption snapshots and overlapping bindings must be exactly equal. Mathematical equivalence
    /// is deliberately insufficient here because the workspace has no typed equality certificate for
    /// a binding conflict. All imported derivations are independently re-verified before mutation.
    pub fn merge(&mut self, incoming: &SemanticWorkspace) -> Result<MergeReceipt, WorkspaceError> {
        validate_branch_name(&self.branch_name)?;
        validate_branch_name(&incoming.branch_name)?;
        if self.bindings.len() > MAX_WORKSPACE_BINDINGS
            || incoming.bindings.len() > MAX_WORKSPACE_BINDINGS
            || self.derivations.len() > MAX_WORKSPACE_DERIVATIONS
            || incoming.derivations.len() > MAX_WORKSPACE_DERIVATIONS
        {
            return Err(WorkspaceError::ResourceLimitExceeded);
        }

        if self.assumptions != incoming.assumptions {
            return Err(WorkspaceError::AssumptionContextMismatch {
                target_context_digest: self.assumptions.digest(),
                incoming_context_digest: incoming.assumptions.digest(),
            });
        }

        for (sym, inc_expr) in &incoming.bindings {
            if let Some(base_expr) = self.bindings.get(sym)
                && base_expr != inc_expr
            {
                return Err(WorkspaceError::BindingConflict {
                    symbol: sym.clone(),
                });
            }
        }

        let mut new_derivations = Vec::new();
        new_derivations
            .try_reserve(incoming.derivations.len())
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        for deriv in &incoming.derivations {
            if fsym_proof_kernel::verify_derivation_independent(deriv, &self.assumptions).is_err() {
                return Err(WorkspaceError::DerivationVerificationFailed);
            }
            if !self.derivations.contains(deriv) && !new_derivations.contains(&deriv) {
                new_derivations.push(deriv);
            }
        }

        let additional_bindings = incoming
            .bindings
            .keys()
            .filter(|symbol| !self.bindings.contains_key(*symbol))
            .count();
        let post_binding_count = self
            .bindings
            .len()
            .checked_add(additional_bindings)
            .ok_or(WorkspaceError::ResourceLimitExceeded)?;
        let post_derivation_count = self
            .derivations
            .len()
            .checked_add(new_derivations.len())
            .ok_or(WorkspaceError::ResourceLimitExceeded)?;
        if post_binding_count > MAX_WORKSPACE_BINDINGS
            || post_derivation_count > MAX_WORKSPACE_DERIVATIONS
        {
            return Err(WorkspaceError::ResourceLimitExceeded);
        }

        let mut post_bindings = Vec::new();
        post_bindings
            .try_reserve(post_binding_count)
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        post_bindings.extend(self.bindings.iter());
        post_bindings.extend(
            incoming
                .bindings
                .iter()
                .filter(|(symbol, _)| !self.bindings.contains_key(*symbol)),
        );
        post_bindings.sort_unstable_by_key(|(left, _)| *left);

        let mut post_derivations = Vec::new();
        post_derivations
            .try_reserve(post_derivation_count)
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        post_derivations.extend(self.derivations.iter());
        post_derivations.extend(new_derivations.iter().copied());

        let content_digest = merge_content_digest(
            &self.branch_name,
            &incoming.branch_name,
            self.assumptions.digest(),
            &post_bindings,
            &post_derivations,
        )?;
        let receipt = MergeReceipt {
            source_branch: incoming.branch_name.clone(),
            target_branch: self.branch_name.clone(),
            merged_bindings_count: post_binding_count,
            content_digest,
        };

        self.bindings
            .try_reserve(additional_bindings)
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        self.derivations
            .try_reserve(new_derivations.len())
            .map_err(|_| WorkspaceError::AllocationFailure)?;
        for (sym, expr) in &incoming.bindings {
            self.bindings.insert(sym.clone(), expr.clone());
        }
        self.derivations
            .extend(new_derivations.into_iter().cloned());

        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_assumptions::Predicate;
    use fsym_proof_kernel::{Claim, DerivationStep, ProofRule, StepId};

    fn reflexive_derivation(expression: Expr) -> DerivationTree {
        DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::Reflexivity(expression.clone()),
                claim: Claim::equality(expression.clone(), expression),
            }],
            root: StepId(0),
        }
    }

    #[test]
    fn aggregate_dag_limit_maps_to_workspace_resource_refusal() {
        assert_eq!(
            map_dag_error(DagError::TotalPayloadLimitExceeded(1)),
            WorkspaceError::ResourceLimitExceeded
        );
    }

    #[test]
    fn receipt_digest_binds_post_merge_content() {
        let mut first_target = SemanticWorkspace::new("main");
        let mut first_source = SemanticWorkspace::new("feature");
        first_source.bind(Symbol::new("x"), Expr::from_i64(1));

        let mut second_target = SemanticWorkspace::new("main");
        let mut second_source = SemanticWorkspace::new("feature");
        second_source.bind(Symbol::new("x"), Expr::from_i64(2));

        let first = first_target.merge(&first_source).unwrap();
        let second = second_target.merge(&second_source).unwrap();

        assert_ne!(first.content_digest, second.content_digest);
    }

    #[test]
    fn receipt_digest_is_independent_of_hash_map_insertion_order() {
        let mut first_target = SemanticWorkspace::new("main");
        first_target.bind(Symbol::new("x"), Expr::from_i64(1));
        first_target.bind(Symbol::new("y"), Expr::from_i64(2));
        let first_source = first_target.fork("feature");

        let mut second_target = SemanticWorkspace::new("main");
        second_target.bind(Symbol::new("y"), Expr::from_i64(2));
        second_target.bind(Symbol::new("x"), Expr::from_i64(1));
        let second_source = second_target.fork("feature");

        let first = first_target.merge(&first_source).unwrap();
        let second = second_target.merge(&second_source).unwrap();

        assert_eq!(first.content_digest, second.content_digest);
    }

    #[test]
    fn merge_refuses_different_assumption_context_without_mutation() {
        let mut target = SemanticWorkspace::new("main");
        target.bind(Symbol::new("stable"), Expr::from_i64(1));
        let before = target.clone();

        let root = ImmutableAssumptionsSnapshot::empty();
        let x = Symbol::new("x");
        let mut facts = HashMap::new();
        facts.insert(x, vec![Predicate::Positive]);
        let mut source = target.fork("feature");
        source.assumptions = root
            .derive_child(facts, HashMap::new(), "feature assumption")
            .unwrap();

        assert!(matches!(
            target.merge(&source),
            Err(WorkspaceError::AssumptionContextMismatch { .. })
        ));
        assert_eq!(target, before);
    }

    #[test]
    fn patch_refuses_ambiguous_edit_without_mutation() {
        let x = Symbol::new("x");
        let mut workspace = SemanticWorkspace::new("main");
        workspace.bind(x.clone(), Expr::from_i64(1));
        let before = workspace.clone();
        let patch = WorkspacePatch {
            updated_bindings: HashMap::from([(x.clone(), Expr::from_i64(2))]),
            removed_bindings: vec![x.clone()],
        };

        assert_eq!(
            workspace.apply_patch(&patch),
            Err(WorkspaceError::AmbiguousBindingEdit { symbol: x })
        );
        assert_eq!(workspace, before);
    }

    #[test]
    fn merge_requires_structural_binding_identity_without_a_certificate() {
        let x = Symbol::new("x");
        let mut target = SemanticWorkspace::new("main");
        target.bind(x.clone(), Expr::from_i64(2));
        let before = target.clone();
        let mut source = target.fork("feature");
        source.bind(
            x.clone(),
            Expr::Add(vec![Expr::from_i64(1), Expr::from_i64(1)]),
        );

        assert_eq!(
            target.merge(&source),
            Err(WorkspaceError::BindingConflict { symbol: x })
        );
        assert_eq!(target, before);
    }

    #[test]
    fn repeated_merge_is_idempotent_for_imported_derivations() {
        let mut target = SemanticWorkspace::new("main");
        let mut source = target.fork("feature");
        source
            .derivations
            .push(reflexive_derivation(Expr::symbol("x")));

        let first = target.merge(&source).unwrap();
        let second = target.merge(&source).unwrap();

        assert_eq!(target.derivations.len(), 1);
        assert_eq!(first.content_digest, second.content_digest);
    }
}
