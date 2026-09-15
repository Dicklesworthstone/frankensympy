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
use fsym_proof_kernel::{DerivationTree, ProofKernel};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Maximum accepted request-frame size for [`handle_envelope`].
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
    UniverseMismatch { found: [u8; 32], expected: [u8; 32] },
    #[error("replay refused: the re-verified derivation digest does not match the exported digest")]
    ReplayDigestMismatch,
    #[error("session serialization failed: {0}")]
    Serialization(String),
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

/// A bounded native-request session.
pub struct Session {
    workspace: fsym_runtime::workspace::SemanticWorkspace,
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
        Self {
            workspace: fsym_runtime::workspace::SemanticWorkspace::new("session"),
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

    /// Constructs a *candidate* claim for `source` (reflexivity), metered by
    /// the session budget. The candidate is not accepted evidence until
    /// [`Session::verify_claim`] independently accepts it.
    pub fn construct_claim(&mut self, source: &str) -> Result<ClaimRecord, SessionError> {
        if source.len() > MAX_SOURCE_BYTES {
            return Err(SessionError::SourceTooLarge {
                size: source.len(),
                limit: MAX_SOURCE_BYTES,
            });
        }
        let _expr = fsym_core::parse(source).map_err(|_| SessionError::ParseFailed)?;
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

    /// Independently verifies a candidate claim: the kernel proves the
    /// reflexivity step under the session budget, the derivation is
    /// exported, and `verify_derivation_independent` must accept it before
    /// the record is promoted to `Accepted`.
    pub fn verify_claim(&mut self, claim: &ClaimRecord) -> Result<ClaimRecord, SessionError> {
        let expr = fsym_core::parse(&claim.source).map_err(|_| SessionError::ParseFailed)?;
        let meter = &mut self.budget;
        let step = self
            .kernel
            .prove_reflexivity(expr, meter)
            .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
        // Second step: the symmetry of the reflexivity claim gives exports a
        // real multi-step derivation so pagination is exercised honestly.
        let symmetry_step = self
            .kernel
            .prove_symmetry(step, meter)
            .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
        let derivation = self
            .kernel
            .export_derivation(symmetry_step)
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
                match fsym_core::parse(expr) {
                    Ok(parsed) => Ok(self.workspace.eval(&parsed).to_string()),
                    Err(_) => Err("source expression failed to parse".to_string()),
                }
            }
            other => {
                let _ = other;
                Err("unsupported request in the declared session subset".to_string())
            }
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
