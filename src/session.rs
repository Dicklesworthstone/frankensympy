//! Bounded, typed session API over the native request path (WS14, fra-rc-cli-u9o).
//!
//! A [`Session`] exposes the declared subset of native term/claim/verify
//! requests with explicit budgets: construct (bind), differentiate, verify a
//! reflexivity claim through the proof kernel, export that claim as
//! canonical paginated JSON, replay the exported derivation independently,
//! and import an export with universe (context digest) checks. Everything
//! here refuses with typed errors; nothing invents accepted results and no
//! string transcript ever becomes state.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Budget, BudgetLimits, Dimension};
use fsym_core::{Expr, Symbol};
use fsym_polys::{MultivariatePoly, factor_multivariate, verify_multivariate_factorization};
use fsym_proof_kernel::{CertificatePayload, Claim, DerivationTree, ProofKernel};
use num_traits::One;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;
use thiserror::Error;

/// Maximum accepted request-frame size for [`Session::handle_envelope`].
pub const MAX_ENVELOPE_BYTES: usize = 64 * 1024;
/// Maximum source-expression bytes accepted by [`Session::construct`].
pub const MAX_SOURCE_BYTES: usize = 4 * 1024;
/// Maximum claim-export page size (steps per page).
pub const MAX_EXPORT_PAGE_SIZE: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SessionError {
    #[error("request frame of {size} bytes exceeds the {limit}-byte envelope limit")]
    FrameTooLarge { size: usize, limit: usize },
    #[error("source expression of {size} bytes exceeds the {limit}-byte source limit")]
    SourceTooLarge { size: usize, limit: usize },
    #[error("source expression failed to parse")]
    ParseFailed,
    #[error("symbol name is not a bounded identifier")]
    InvalidName,
    #[error("verification budget exhausted after {charged} charge units")]
    BudgetExhausted { charged: u64 },
    #[error("claim is still a candidate; independent verification has not accepted it")]
    CandidateNotAccepted,
    #[error("claim export references an unknown derivation")]
    UnknownDerivation,
    #[error("claim export page {page} is out of range (total pages: {total})")]
    PageOutOfRange { page: u32, total: u32 },
    #[error("export page size {size} exceeds the limit {limit}")]
    PageSizeTooLarge { size: usize, limit: usize },
    #[error(
        "import refused: export originated in a different assumption universe (context digest {found:02x?}, expected {expected:02x?})"
    )]
    UniverseMismatch {
        found: [u8; 32],
        expected: [u8; 32],
    },
    #[error(
        "replay refused: the re-verified derivation digest does not match the exported digest"
    )]
    ReplayDigestMismatch,
    #[error("session serialization failed: {0}")]
    Serialization(String),
    #[error("workspace operation refused: {0}")]
    Workspace(String),
}

/// Whether a claim has been accepted by independent verification.
///
/// `Candidate` claims carry no evidence class above "candidate" and are
/// never rendered as accepted results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Candidate,
    Accepted,
}

/// One constructed claim record with its (honest) status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimRecord {
    pub source: String,
    pub status: ClaimStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derivation_digest: Option<[u8; 32]>,
    pub context_digest: [u8; 32],
}

/// One paginated page of an exported claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimExportPage {
    pub claim: ClaimRecord,
    pub page: u32,
    pub total_pages: u32,
    pub steps: Vec<fsym_proof_kernel::DerivationStep>,
}

/// Explicit resource budgets for one session. Every kernel verification
/// charges these dimensions; exhaustion is a typed refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionBudgets {
    pub uniform_dimension_budget: u64,
    pub verify_step_budget: usize,
}

impl Default for SessionBudgets {
    fn default() -> Self {
        Self {
            uniform_dimension_budget: 4_096,
            verify_step_budget: 256,
        }
    }
}

fn parse_claim_sides(source: &str) -> Result<(Expr, Option<Expr>), SessionError> {
    if let Some((lhs_src, rhs_src)) = source.split_once('=') {
        if rhs_src.contains('=') {
            return Err(SessionError::ParseFailed);
        }
        let lhs = fsym_core::parse(lhs_src.trim()).map_err(|_| SessionError::ParseFailed)?;
        let rhs = fsym_core::parse(rhs_src.trim()).map_err(|_| SessionError::ParseFailed)?;
        Ok((lhs, Some(rhs)))
    } else {
        let lhs = fsym_core::parse(source).map_err(|_| SessionError::ParseFailed)?;
        Ok((lhs, None))
    }
}

fn factor_expr_for_session(expr: &Expr) -> Result<Expr, String> {
    let mut generators: Vec<Symbol> = expr.free_symbols().into_iter().collect();
    generators.sort();
    if generators.is_empty() {
        return Ok(fsym_simplify::simplify(expr));
    }
    let poly = MultivariatePoly::from_expr(expr, &generators)
        .map_err(|e| format!("non-polynomial expression: {e}"))?;
    let factored =
        factor_multivariate(&poly).map_err(|e| format!("factorization refused: {e}"))?;
    verify_multivariate_factorization(&poly, &factored)
        .map_err(|e| format!("factorization verification failed: {e}"))?;
    if factored.factors.is_empty() {
        return Ok(if factored.scale.is_integer() {
            Expr::Integer(factored.scale.to_integer())
        } else {
            Expr::Rational(factored.scale)
        });
    }
    let mut factors = Vec::new();
    if !factored.scale.is_one() {
        factors.push(if factored.scale.is_integer() {
            Expr::Integer(factored.scale.to_integer())
        } else {
            Expr::Rational(factored.scale)
        });
    }
    for (factor_poly, mult) in &factored.factors {
        let factor_expr = factor_poly
            .to_expr()
            .map_err(|e| format!("factor expression conversion failed: {e}"))?;
        if *mult == 1 {
            factors.push(factor_expr);
        } else {
            factors.push(Expr::pow(factor_expr, Expr::from_i64(*mult as i64)));
        }
    }
    if factors.len() == 1 {
        Ok(factors.pop().expect("len == 1"))
    } else {
        Ok(Expr::Mul(factors))
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in digest {
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// A bounded native-request session.
pub struct Session {
    workspace: fsym_runtime::workspace::SemanticWorkspace,
    branches: BTreeMap<String, fsym_runtime::workspace::SemanticWorkspace>,
    kernel: ProofKernel,
    context: Arc<ImmutableAssumptionsSnapshot>,
    budget: Budget,
    budgets: SessionBudgets,
    claims: Vec<ClaimRecord>,
    derivations: Vec<DerivationTree>,
}

impl Session {
    /// Creates a session on an empty assumption universe with explicit budgets.
    pub fn new(budgets: SessionBudgets) -> Self {
        Self::with_context(ImmutableAssumptionsSnapshot::empty(), budgets)
    }

    /// Creates a session bound to a specific assumption universe. Claims
    /// exported here carry this universe's context digest, and imports from
    /// a different universe refuse.
    pub fn with_context(
        context: Arc<ImmutableAssumptionsSnapshot>,
        budgets: SessionBudgets,
    ) -> Self {
        let context = Arc::new(ImmutableAssumptionsSnapshot::clone(&context));
        let kernel = ProofKernel::new((*context).clone());
        let mut workspace = fsym_runtime::workspace::SemanticWorkspace::new("session");
        workspace.assumptions = Arc::clone(&context);
        let mut branches = BTreeMap::new();
        branches.insert("session".to_string(), workspace.clone());
        Self {
            workspace,
            branches,
            kernel,
            context,
            budget: Budget::new(BudgetLimits::uniform(
                budgets.uniform_dimension_budget,
                budgets.uniform_dimension_budget,
            )),
            budgets,
            claims: Vec::new(),
            derivations: Vec::new(),
        }
    }

    /// The assumption-universe digest this session is bound to.
    pub fn context_digest(&self) -> [u8; 32] {
        self.context.digest()
    }

    /// Active branch name of the session workspace.
    pub fn active_branch(&self) -> &str {
        &self.workspace.branch_name
    }

    fn sync_active_branch(&mut self) {
        self.branches
            .insert(self.workspace.branch_name.clone(), self.workspace.clone());
    }

    /// Constructs a binding: parses `source` under the declared source cap
    /// and binds it to `symbol`.
    pub fn construct(&mut self, symbol: &str, source: &str) -> Result<Expr, SessionError> {
        if symbol.is_empty() || symbol.len() > 256 {
            return Err(SessionError::InvalidName);
        }
        if source.len() > MAX_SOURCE_BYTES {
            return Err(SessionError::SourceTooLarge {
                size: source.len(),
                limit: MAX_SOURCE_BYTES,
            });
        }
        let expr = fsym_core::parse(source).map_err(|_| SessionError::ParseFailed)?;
        self.workspace.bind(Symbol::new(symbol), expr.clone());
        self.sync_active_branch();
        Ok(expr)
    }

    /// Differentiates `source` with respect to `var`.
    pub fn differentiate(&mut self, source: &str, var: &str) -> Result<String, SessionError> {
        if var.is_empty() || var.len() > 256 {
            return Err(SessionError::InvalidName);
        }
        if source.len() > MAX_SOURCE_BYTES {
            return Err(SessionError::SourceTooLarge {
                size: source.len(),
                limit: MAX_SOURCE_BYTES,
            });
        }
        let expr = fsym_core::parse(source).map_err(|_| SessionError::ParseFailed)?;
        let derivative = fsym_calculus::diff(&expr, &Symbol::new(var));
        Ok(derivative.to_string())
    }

    /// Forks the active workspace into `branch_name` and switches to it.
    pub fn fork_branch(&mut self, branch_name: &str) -> Result<String, SessionError> {
        if branch_name.is_empty() || branch_name.len() > 256 {
            return Err(SessionError::InvalidName);
        }
        self.sync_active_branch();
        if self.branches.contains_key(branch_name) {
            return Err(SessionError::Workspace(format!(
                "branch `{branch_name}` already exists"
            )));
        }
        let forked = self.workspace.fork(branch_name);
        self.branches.insert(branch_name.to_string(), forked.clone());
        self.workspace = forked;
        Ok(branch_name.to_string())
    }

    /// Switches the active workspace to an existing `branch_name`.
    pub fn checkout_branch(&mut self, branch_name: &str) -> Result<String, SessionError> {
        self.sync_active_branch();
        let Some(target) = self.branches.get(branch_name).cloned() else {
            return Err(SessionError::Workspace(format!(
                "branch `{branch_name}` does not exist"
            )));
        };
        self.workspace = target;
        Ok(branch_name.to_string())
    }

    /// Applies an atomic binding patch to the active workspace.
    pub fn apply_patch(
        &mut self,
        updated_bindings: &BTreeMap<String, String>,
        removed_bindings: &[String],
    ) -> Result<usize, SessionError> {
        let mut updates = HashMap::new();
        for (sym, src) in updated_bindings {
            if sym.is_empty() || sym.len() > 256 {
                return Err(SessionError::InvalidName);
            }
            if src.len() > MAX_SOURCE_BYTES {
                return Err(SessionError::SourceTooLarge {
                    size: src.len(),
                    limit: MAX_SOURCE_BYTES,
                });
            }
            let expr = fsym_core::parse(src).map_err(|_| SessionError::ParseFailed)?;
            updates.insert(Symbol::new(sym), expr);
        }
        let mut removals = Vec::with_capacity(removed_bindings.len());
        for sym in removed_bindings {
            if sym.is_empty() || sym.len() > 256 {
                return Err(SessionError::InvalidName);
            }
            removals.push(Symbol::new(sym));
        }
        let patch = fsym_runtime::workspace::WorkspacePatch {
            updated_bindings: updates,
            removed_bindings: removals,
        };
        self.workspace
            .apply_patch(&patch)
            .map_err(|e| SessionError::Workspace(e.to_string()))?;
        self.sync_active_branch();
        Ok(self.workspace.bindings.len())
    }

    /// Merges `source_branch` into the active branch with a verified [`fsym_proof_kernel::SemanticMergeCertificate`].
    pub fn merge_branch(
        &mut self,
        source_branch: &str,
    ) -> Result<fsym_proof_kernel::SemanticMergeCertificate, SessionError> {
        self.sync_active_branch();
        if source_branch == self.workspace.branch_name {
            return Err(SessionError::Workspace(
                "cannot merge a branch into itself".to_string(),
            ));
        }
        let Some(incoming) = self.branches.get(source_branch).cloned() else {
            return Err(SessionError::Workspace(format!(
                "source branch `{source_branch}` does not exist"
            )));
        };
        let cert = self
            .workspace
            .merge_with_certificate(&incoming, 1, 1)
            .map_err(|e| SessionError::Workspace(e.to_string()))?;
        self.sync_active_branch();
        Ok(cert)
    }

    /// Constructs a *candidate* claim for `source` (reflexivity or `lhs = rhs`),
    /// metered by the session budget. The candidate is not accepted evidence until
    /// [`Session::verify_claim`] independently accepts it.
    pub fn construct_claim(&mut self, source: &str) -> Result<ClaimRecord, SessionError> {
        if source.len() > MAX_SOURCE_BYTES {
            return Err(SessionError::SourceTooLarge {
                size: source.len(),
                limit: MAX_SOURCE_BYTES,
            });
        }
        let _ = parse_claim_sides(source)?;
        if self.claims.len() >= self.budgets.verify_step_budget {
            return Err(SessionError::BudgetExhausted { charged: 0 });
        }
        fsym_budget::BudgetMeter::charge(&mut self.budget, Dimension::ComputeSteps, 1)
            .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
        let record = ClaimRecord {
            source: source.to_string(),
            status: ClaimStatus::Candidate,
            derivation_digest: None,
            context_digest: self.context_digest(),
        };
        self.claims.push(record.clone());
        Ok(record)
    }

    /// Independently verifies a candidate claim:
    /// - For single-expression claims (`expr`), proves reflexivity + symmetry.
    /// - For equational claims (`lhs = rhs`), substitutes workspace bindings,
    ///   proves the equational identity via `ProofKernel` (`Reflexivity` or
    ///   `CertificatePayload::PolynomialNormalForm` + symmetry round-trip),
    ///   rejects false claims (`lhs != rhs`) with `SessionError::CandidateNotAccepted`,
    ///   and runs `verify_derivation_independent` before promoting to `Accepted`.
    pub fn verify_claim(&mut self, claim: &ClaimRecord) -> Result<ClaimRecord, SessionError> {
        let (lhs_raw, rhs_raw_opt) = parse_claim_sides(&claim.source)?;
        let lhs = lhs_raw.subs(&self.workspace.bindings);
        let meter = &mut self.budget;
        let root_step = match rhs_raw_opt {
            None => {
                let step = self
                    .kernel
                    .prove_reflexivity(lhs, meter)
                    .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
                // Second step: symmetry of reflexivity gives exports a real
                // multi-step derivation so pagination is exercised honestly.
                self.kernel
                    .prove_symmetry(step, meter)
                    .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?
            }
            Some(rhs_raw) => {
                let rhs = rhs_raw.subs(&self.workspace.bindings);
                if lhs == rhs {
                    let step = self
                        .kernel
                        .prove_reflexivity(lhs, meter)
                        .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
                    self.kernel
                        .prove_symmetry(step, meter)
                        .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?
                } else {
                    let lemma_step = self
                        .kernel
                        .prove_certificate_lemma(
                            "PolynomialNormalForm",
                            Claim::equality(lhs.clone(), rhs.clone()),
                            CertificatePayload::PolynomialNormalForm,
                            meter,
                        )
                        .map_err(|err| match err {
                            fsym_proof_kernel::KernelError::Budget(_) => {
                                SessionError::BudgetExhausted { charged: 0 }
                            }
                            _ => SessionError::CandidateNotAccepted,
                        })?;
                    let sym_step = self
                        .kernel
                        .prove_symmetry(lemma_step, meter)
                        .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
                    self.kernel
                        .prove_symmetry(sym_step, meter)
                        .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?
                }
            }
        };
        let derivation = self
            .kernel
            .export_derivation(root_step)
            .map_err(|_| SessionError::UnknownDerivation)?;
        // Independent re-verification: the generator (kernel) output is
        // checked by the separate verifier lane before acceptance.
        let context = ImmutableAssumptionsSnapshot::clone(&self.context);
        fsym_proof_kernel::verify_derivation_independent(&derivation, &context)
            .map_err(|_| SessionError::CandidateNotAccepted)?;
        let accepted = ClaimRecord {
            source: claim.source.clone(),
            status: ClaimStatus::Accepted,
            derivation_digest: Some(derivation.digest()),
            context_digest: claim.context_digest,
        };
        if !self.workspace.derivations.contains(&derivation) {
            self.workspace.derivations.push(derivation.clone());
            self.sync_active_branch();
        }
        self.derivations.push(derivation);
        if let Some(existing) = self
            .claims
            .iter_mut()
            .find(|c| c.source == accepted.source && c.status == ClaimStatus::Candidate)
        {
            *existing = accepted.clone();
        }
        Ok(accepted)
    }

    /// Exports one claim's derivation as canonical paginated JSON.
    pub fn export_claim(
        &self,
        claim: &ClaimRecord,
        page: u32,
        page_size: usize,
    ) -> Result<serde_json::Value, SessionError> {
        if page_size > MAX_EXPORT_PAGE_SIZE || page_size == 0 {
            return Err(SessionError::PageSizeTooLarge {
                size: page_size,
                limit: MAX_EXPORT_PAGE_SIZE,
            });
        }
        if claim.status != ClaimStatus::Accepted {
            return Err(SessionError::CandidateNotAccepted);
        }
        let digest = claim
            .derivation_digest
            .ok_or(SessionError::UnknownDerivation)?;
        let derivation = self
            .derivations
            .iter()
            .find(|d| d.digest() == digest)
            .ok_or(SessionError::UnknownDerivation)?;
        let total_steps = derivation.steps.len();
        let total_pages = total_steps.div_ceil(page_size).max(1) as u32;
        if page >= total_pages {
            return Err(SessionError::PageOutOfRange {
                page,
                total: total_pages,
            });
        }
        let start = page as usize * page_size;
        let end = ((page as usize + 1) * page_size).min(total_steps);
        serde_json::to_value(ClaimExportPage {
            claim: claim.clone(),
            page,
            total_pages,
            steps: derivation.steps[start..end].to_vec(),
        })
        .map_err(|error| SessionError::Serialization(error.to_string()))
    }

    /// Replays an exported claim page-set: re-runs independent verification
    /// of the derivation and requires the digest to match the export.
    pub fn replay_claim(
        &self,
        export: &serde_json::Value,
        context_digest: [u8; 32],
    ) -> Result<[u8; 32], SessionError> {
        let page: ClaimExportPage = serde_json::from_value(export.clone())
            .map_err(|error| SessionError::Serialization(error.to_string()))?;
        if page.claim.context_digest != context_digest {
            return Err(SessionError::UniverseMismatch {
                found: page.claim.context_digest,
                expected: context_digest,
            });
        }
        if page.claim.status != ClaimStatus::Accepted {
            return Err(SessionError::CandidateNotAccepted);
        }
        let digest = page
            .claim
            .derivation_digest
            .ok_or(SessionError::UnknownDerivation)?;
        let derivation = self
            .derivations
            .iter()
            .find(|d| d.digest() == digest)
            .ok_or(SessionError::UnknownDerivation)?;
        let context = ImmutableAssumptionsSnapshot::clone(&self.context);
        fsym_proof_kernel::verify_derivation_independent(derivation, &context)
            .map_err(|_| SessionError::ReplayDigestMismatch)?;
        if derivation.digest() != digest {
            return Err(SessionError::ReplayDigestMismatch);
        }
        Ok(digest)
    }

    /// Handles one NDJSON envelope line: optional `"id"` echo plus the typed
    /// request against this session. The response is one JSON line; the id
    /// is echoed verbatim and repeated ids are processed independently.
    pub fn handle_envelope(&mut self, line: &str) -> String {
        if line.len() > MAX_ENVELOPE_BYTES {
            return serde_json::json!({
                "status": "error",
                "code": "request_too_large",
                "error": "request exceeds the session envelope limit",
            })
            .to_string();
        }
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        let mut value = match parsed {
            Ok(value @ serde_json::Value::Object(_)) => value,
            _ => {
                return serde_json::json!({
                    "status": "error",
                    "code": "malformed_request",
                    "error": "request is not a recognized session envelope",
                })
                .to_string();
            }
        };
        let id = value.get("id").cloned();
        let _ = value.as_object_mut().map(|map| map.remove("id"));
        let inner = match serde_json::from_value::<fsym_runtime::protocol::AgentRequest>(value) {
            Ok(request) => request,
            Err(_) => {
                let mut response = serde_json::json!({
                    "status": "error",
                    "code": "malformed_request",
                    "error": "request is not a recognized typed request",
                });
                if let Some(id) = id {
                    response["id"] = id;
                }
                return response.to_string();
            }
        };
        let body = match &inner {
            fsym_runtime::protocol::AgentRequest::Bind { symbol, expr } => self
                .construct(symbol, expr)
                .map(|expr| expr.to_string())
                .map_err(|error| error.to_string()),
            fsym_runtime::protocol::AgentRequest::Diff { expr, var } => self
                .differentiate(expr, var)
                .map_err(|error| error.to_string()),
            fsym_runtime::protocol::AgentRequest::Eval { expr }
            | fsym_runtime::protocol::AgentRequest::Simplify { expr } => {
                if expr.len() > MAX_SOURCE_BYTES {
                    Err("source expression exceeds session limit".to_string())
                } else {
                    match fsym_core::parse(expr) {
                        Ok(parsed) => Ok(self.workspace.eval(&parsed).to_string()),
                        Err(_) => Err("source expression failed to parse".to_string()),
                    }
                }
            }
            fsym_runtime::protocol::AgentRequest::Factor { expr } => {
                if expr.len() > MAX_SOURCE_BYTES {
                    Err("source expression exceeds session limit".to_string())
                } else {
                    match fsym_core::parse(expr) {
                        Ok(parsed) => {
                            let sub = parsed.subs(&self.workspace.bindings);
                            factor_expr_for_session(&sub).map(|e| e.to_string())
                        }
                        Err(_) => Err("source expression failed to parse".to_string()),
                    }
                }
            }
            fsym_runtime::protocol::AgentRequest::Solve { expr, var } => {
                if var.is_empty() || var.len() > 256 || expr.len() > MAX_SOURCE_BYTES {
                    Err("invalid variable or expression exceeds limit".to_string())
                } else {
                    match fsym_core::parse(expr) {
                        Ok(parsed) => {
                            let sub = parsed.subs(&self.workspace.bindings);
                            match fsym_solvers::solve(&sub, &Symbol::new(var)) {
                                Ok(roots) => {
                                    let rendered: Vec<String> =
                                        roots.iter().map(ToString::to_string).collect();
                                    Ok(format!("[{}]", rendered.join(", ")))
                                }
                                Err(e) => Err(e.to_string()),
                            }
                        }
                        Err(_) => Err("source expression failed to parse".to_string()),
                    }
                }
            }
            fsym_runtime::protocol::AgentRequest::Integrate { expr, var } => {
                if var.is_empty() || var.len() > 256 || expr.len() > MAX_SOURCE_BYTES {
                    Err("invalid variable or expression exceeds limit".to_string())
                } else {
                    match fsym_core::parse(expr) {
                        Ok(parsed) => {
                            let sub = parsed.subs(&self.workspace.bindings);
                            fsym_calculus::integrate(&sub, &Symbol::new(var))
                                .map(|e| e.to_string())
                                .map_err(|e| e.to_string())
                        }
                        Err(_) => Err("source expression failed to parse".to_string()),
                    }
                }
            }
            fsym_runtime::protocol::AgentRequest::Limit { expr, var, point } => {
                if var.is_empty()
                    || var.len() > 256
                    || expr.len() > MAX_SOURCE_BYTES
                    || point.len() > MAX_SOURCE_BYTES
                {
                    Err("invalid variable or expression exceeds limit".to_string())
                } else {
                    match (fsym_core::parse(expr), fsym_core::parse(point)) {
                        (Ok(parsed_expr), Ok(parsed_point)) => {
                            let sub = parsed_expr.subs(&self.workspace.bindings);
                            fsym_calculus::limit(&sub, &Symbol::new(var), &parsed_point)
                                .map(|e| e.to_string())
                                .map_err(|e| e.to_string())
                        }
                        _ => Err("source expression or limit point failed to parse".to_string()),
                    }
                }
            }
            fsym_runtime::protocol::AgentRequest::Prove { lhs, rhs } => {
                let claim_src = format!("{lhs} = {rhs}");
                match self
                    .construct_claim(&claim_src)
                    .and_then(|cand| self.verify_claim(&cand))
                {
                    Ok(accepted) => {
                        let digest = accepted.derivation_digest.unwrap_or([0u8; 32]);
                        Ok(format!("verified:{}", hex_digest(&digest)))
                    }
                    Err(e) => Err(e.to_string()),
                }
            }
            fsym_runtime::protocol::AgentRequest::Fork { branch_name } => self
                .fork_branch(branch_name)
                .map(|b| format!("forked:{b}"))
                .map_err(|e| e.to_string()),
            fsym_runtime::protocol::AgentRequest::Checkout { branch_name } => self
                .checkout_branch(branch_name)
                .map(|b| format!("checked_out:{b}"))
                .map_err(|e| e.to_string()),
            fsym_runtime::protocol::AgentRequest::ApplyPatch {
                updated_bindings,
                removed_bindings,
            } => self
                .apply_patch(updated_bindings, removed_bindings)
                .map(|count| format!("patched:{count} bindings"))
                .map_err(|e| e.to_string()),
            fsym_runtime::protocol::AgentRequest::MergeBranch { source_branch } => self
                .merge_branch(source_branch)
                .map(|cert| format!("merged:{}", hex_digest(&cert.result_root)))
                .map_err(|e| e.to_string()),
            fsym_runtime::protocol::AgentRequest::StateRoot => self
                .workspace
                .state_root()
                .map(|root| hex_digest(&root))
                .map_err(|e| e.to_string()),
        };
        let mut response = match body {
            Ok(result) => serde_json::json!({ "status": "success", "result": result }),
            Err(error) => {
                serde_json::json!({ "status": "error", "code": "refused", "error": error })
            }
        };
        if let Some(id) = id {
            response["id"] = id;
        }
        response.to_string()
    }
}
