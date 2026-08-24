//! Semantic workspaces, branch forking, patching, and verifier-checked merge (WS14 / architecture §7.8).

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::DerivationTree;
use fsym_simplify::simplify;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkspaceError {
    #[error(
        "Semantic merge conflict: symbol '{0}' bound to conflicting expressions '{1}' and '{2}'"
    )]
    BindingConflict(String, String, String),
    #[error("Semantic merge conflict: assumption conflict detected on symbol '{0}'")]
    AssumptionConflict(String),
    #[error("Invalid patch: {0}")]
    InvalidPatch(String),
    #[error("Derivation verification failed for incoming branch")]
    DerivationVerificationFailed,
}

/// A semantic patch modifying workspace bindings or assumptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspacePatch {
    pub updated_bindings: HashMap<Symbol, Expr>,
    pub removed_bindings: Vec<Symbol>,
}

/// Verification receipt for a clean semantic merge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeReceipt {
    pub source_branch: String,
    pub target_branch: String,
    pub merged_bindings_count: usize,
    pub content_digest: [u8; 32],
}

/// Isolated semantic workspace holding bindings, assumptions context, and derivations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticWorkspace {
    pub branch_name: String,
    pub bindings: HashMap<Symbol, Expr>,
    #[serde(skip, default = "default_assumptions")]
    pub assumptions: Arc<ImmutableAssumptionsSnapshot>,
    pub derivations: Vec<DerivationTree>,
}

fn default_assumptions() -> Arc<ImmutableAssumptionsSnapshot> {
    ImmutableAssumptionsSnapshot::empty()
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
        for (sym, expr) in &patch.updated_bindings {
            self.bindings.insert(sym.clone(), expr.clone());
        }
        for sym in &patch.removed_bindings {
            self.bindings.remove(sym);
        }
        Ok(())
    }

    /// Merges an incoming branch into `self`, requiring independent verifier check for semantic conflicts.
    pub fn merge(&mut self, incoming: &SemanticWorkspace) -> Result<MergeReceipt, WorkspaceError> {
        // Check for semantic conflicts in overlapping bindings
        for (sym, inc_expr) in &incoming.bindings {
            if let Some(base_expr) = self.bindings.get(sym) {
                let diff = simplify(&Expr::Add(vec![
                    base_expr.clone(),
                    Expr::Mul(vec![Expr::from_i64(-1), inc_expr.clone()]),
                ]));
                if !diff.is_zero() && base_expr != inc_expr {
                    return Err(WorkspaceError::BindingConflict(
                        sym.name.clone(),
                        base_expr.to_string(),
                        inc_expr.to_string(),
                    ));
                }
            }
        }

        // Check and independently verify incoming derivations
        for deriv in &incoming.derivations {
            if fsym_proof_kernel::verify_derivation_independent(deriv, &self.assumptions).is_err() {
                return Err(WorkspaceError::DerivationVerificationFailed);
            }
        }

        // Apply incoming bindings
        for (sym, expr) in &incoming.bindings {
            self.bindings.insert(sym.clone(), expr.clone());
        }

        // Merge derivations
        self.derivations.extend(incoming.derivations.clone());

        // Compute merge receipt digest
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.workspace.merge.v1:");
        hasher.update(self.branch_name.as_bytes());
        hasher.update(incoming.branch_name.as_bytes());
        hasher.update(&self.bindings.len().to_le_bytes());
        let content_digest = *hasher.finalize().as_bytes();

        Ok(MergeReceipt {
            source_branch: incoming.branch_name.clone(),
            target_branch: self.branch_name.clone(),
            merged_bindings_count: self.bindings.len(),
            content_digest,
        })
    }
}
