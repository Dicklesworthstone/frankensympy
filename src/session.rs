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
use fsym_budget::{Budget, BudgetLimits, Dimension, VerifierLease};
use fsym_core::{Expr, Symbol};
use fsym_polys::{MultivariatePoly, factor_multivariate, verify_multivariate_factorization};
use fsym_proof_kernel::{CertificatePayload, Claim, DerivationTree, ProofKernel, StepId};
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
/// Maximum encoded size of one complete portable claim export.
pub const MAX_CLAIM_EXPORT_BYTES: usize = 1024 * 1024;
const MAX_CLAIM_BINDINGS: usize = 128;
const CLAIM_EXPORT_SCHEMA: u32 = 1;

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
    #[error("claim metadata does not describe the independently verified statement")]
    ClaimMismatch,
    #[error("claim export pages are incomplete, inconsistent, or out of order")]
    InvalidPages,
    #[error("claim export exceeds its bounded transport or verification limit")]
    ExportLimit,
    #[error("session claim substitution under a binder is unsupported")]
    UnsupportedSubstitution,
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
    /// The exact equality frozen when this candidate was constructed.
    pub statement: Claim,
    /// Only source-referenced bindings, in canonical symbol order. These are
    /// substitution inputs, not assumptions or evidence for the equality.
    pub bindings: Vec<(Symbol, Expr)>,
    pub status: ClaimStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derivation_digest: Option<[u8; 32]>,
    pub context_digest: [u8; 32],
}

/// One paginated page of an exported claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimExportPage {
    pub schema_version: u32,
    pub claim: ClaimRecord,
    pub root: StepId,
    pub page: u32,
    pub total_pages: u32,
    pub steps: Vec<fsym_proof_kernel::DerivationStep>,
}

/// Structured claim operations of the session facade. Mathematical operations
/// continue to use the existing `AgentRequest` schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", deny_unknown_fields)]
pub enum ClaimRequest {
    ConstructClaim {
        source: String,
    },
    VerifyClaim {
        claim: ClaimRecord,
    },
    ExportClaim {
        claim: ClaimRecord,
        page: u32,
        page_size: usize,
    },
    ReplayClaim {
        pages: Vec<ClaimExportPage>,
    },
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

fn preflight_claim_record(record: &ClaimRecord) -> Result<(), SessionError> {
    if record.source.len() > MAX_SOURCE_BYTES {
        return Err(SessionError::SourceTooLarge {
            size: record.source.len(),
            limit: MAX_SOURCE_BYTES,
        });
    }
    if record.bindings.len() > MAX_CLAIM_BINDINGS {
        return Err(SessionError::ExportLimit);
    }
    fsym_proof_kernel::claim_verification_units(&record.statement)
        .map_err(|_| SessionError::ExportLimit)?;
    for (_, value) in &record.bindings {
        fsym_proof_kernel::expression_verification_units(value)
            .map_err(|_| SessionError::ExportLimit)?;
    }
    bounded_serialized_size(record)?;
    Ok(())
}

fn bounded_substitution(
    lhs: &Expr,
    rhs: &Expr,
    bindings: &HashMap<Symbol, Expr>,
) -> Result<(Expr, Expr), SessionError> {
    let units = |expr: &Expr| {
        fsym_proof_kernel::expression_verification_units(expr)
            .map_err(|_| SessionError::ExportLimit)
    };
    // Each verifier unit upper-bounds 64 structural/payload work units. Refuse
    // potential expansion *before* subs clones a large binding at every use.
    let source_units = units(lhs)?.saturating_add(units(rhs)?).saturating_mul(64);
    let mut replacement_units = 1;
    for value in bindings.values() {
        replacement_units = replacement_units.max(units(value)?.saturating_mul(64));
    }
    if source_units.saturating_mul(replacement_units) > 262_144 {
        return Err(SessionError::ExportLimit);
    }
    // Claims are structural objects. In particular, evaluating a small nested
    // power here could allocate billions of bits before a verifier sees it.
    // Retain raw constructors and let the bounded kernel prove the equality.
    fn substitute(expr: &Expr, bindings: &HashMap<Symbol, Expr>) -> Result<Expr, SessionError> {
        Ok(match expr {
            Expr::Sym(symbol) => bindings.get(symbol).unwrap_or(expr).clone(),
            Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => expr.clone(),
            Expr::Add(terms) => Expr::Add(
                terms
                    .iter()
                    .map(|t| substitute(t, bindings))
                    .collect::<Result<_, _>>()?,
            ),
            Expr::Mul(terms) => Expr::Mul(
                terms
                    .iter()
                    .map(|t| substitute(t, bindings))
                    .collect::<Result<_, _>>()?,
            ),
            Expr::Pow(base, exponent) => Expr::Pow(
                Arc::new(substitute(base, bindings)?),
                Arc::new(substitute(exponent, bindings)?),
            ),
            Expr::Function(name, args) => {
                if matches!(
                    name.as_str(),
                    "Lambda" | "Sum" | "Product" | "Integral" | "Subs" | "Derivative" | "diff"
                ) {
                    if expr.free_symbols().iter().any(|s| bindings.contains_key(s)) {
                        // This bounded transport slice does not invent binder
                        // semantics or call the eager capture-avoiding evaluator.
                        return Err(SessionError::UnsupportedSubstitution);
                    }
                    expr.clone()
                } else {
                    Expr::Function(
                        name.clone(),
                        args.iter()
                            .map(|a| substitute(a, bindings))
                            .collect::<Result<_, _>>()?,
                    )
                }
            }
        })
    }
    Ok((substitute(lhs, bindings)?, substitute(rhs, bindings)?))
}

fn checked_statement(record: &ClaimRecord) -> Result<&Claim, SessionError> {
    preflight_claim_record(record)?;
    let (lhs, rhs) = parse_claim_sides(&record.source)?;
    let rhs = rhs.unwrap_or_else(|| lhs.clone());
    let symbols: std::collections::BTreeSet<_> = lhs
        .free_symbols()
        .into_iter()
        .chain(rhs.free_symbols())
        .collect();
    let mut previous = None;
    for (symbol, _) in &record.bindings {
        if !symbols.contains(symbol) || previous.is_some_and(|p| p >= symbol) {
            return Err(SessionError::ClaimMismatch);
        }
        previous = Some(symbol);
    }
    let bindings: HashMap<_, _> = record.bindings.iter().cloned().collect();
    let (lhs, rhs) = bounded_substitution(&lhs, &rhs, &bindings)?;
    let expected = Claim::equality(lhs, rhs);
    fsym_proof_kernel::claim_verification_units(&expected)
        .map_err(|_| SessionError::ExportLimit)?;
    if expected != record.statement {
        return Err(SessionError::ClaimMismatch);
    }
    Ok(&record.statement)
}

/// Count encoded bytes without first allocating an unbounded serialized copy.
fn bounded_serialized_size(value: &impl Serialize) -> Result<usize, SessionError> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("claim export length overflow"))?;
            if self.0 > MAX_CLAIM_EXPORT_BYTES {
                return Err(std::io::Error::other("claim export size limit"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value).map_err(|_| SessionError::ExportLimit)?;
    Ok(counter.0)
}

fn preflight_export_json(value: &serde_json::Value) -> Result<(), SessionError> {
    let mut pending = vec![(value, 0usize)];
    let mut visited = 0usize;
    while let Some((value, depth)) = pending.pop() {
        visited += 1;
        if depth > 64 || visited + pending.len() > 32_768 {
            return Err(SessionError::ExportLimit);
        }
        match value {
            serde_json::Value::Array(values) => {
                if visited + pending.len() + values.len() > 32_768 {
                    return Err(SessionError::ExportLimit);
                }
                pending.extend(values.iter().map(|v| (v, depth + 1)));
            }
            serde_json::Value::Object(values) => {
                if visited + pending.len() + values.len() > 32_768 {
                    return Err(SessionError::ExportLimit);
                }
                pending.extend(values.values().map(|v| (v, depth + 1)));
            }
            _ => {}
        }
    }
    bounded_serialized_size(value)?;
    Ok(())
}

/// Require the complete schema, including nested proof and expression fields.
/// Some shared native types accept unknown serde fields for other consumers;
/// the portable claim boundary must not silently discard supplied metadata.
fn decode_claim_json<T>(value: &serde_json::Value) -> Result<T, SessionError>
where
    T: serde::de::DeserializeOwned + Serialize,
{
    preflight_export_json(value)?;
    let decoded: T = serde_json::from_value(value.clone())
        .map_err(|error| SessionError::Serialization(error.to_string()))?;
    bounded_serialized_size(&decoded)?;
    let canonical = serde_json::to_value(&decoded)
        .map_err(|error| SessionError::Serialization(error.to_string()))?;
    if canonical != *value {
        return Err(SessionError::Serialization(
            "claim transport contains unknown fields or noncanonical payloads".to_string(),
        ));
    }
    Ok(decoded)
}

fn factor_expr_for_session(expr: &Expr) -> Result<Expr, String> {
    let mut generators: Vec<Symbol> = expr.free_symbols().into_iter().collect();
    generators.sort();
    if generators.is_empty() {
        return Ok(fsym_simplify::simplify(expr));
    }
    let poly = MultivariatePoly::from_expr(expr, &generators)
        .map_err(|e| format!("non-polynomial expression: {e}"))?;
    let factored = factor_multivariate(&poly).map_err(|e| format!("factorization refused: {e}"))?;
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
    verifier_lease: Option<VerifierLease>,
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
        let mut budget = Budget::new(BudgetLimits::uniform(
            budgets.uniform_dimension_budget,
            budgets.uniform_dimension_budget,
        ));
        let verifier_lease = budget.verifier_lease();
        Self {
            workspace,
            branches,
            kernel,
            context,
            budget,
            verifier_lease,
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
        self.branches
            .insert(branch_name.to_string(), forked.clone());
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
        let (lhs, rhs) = parse_claim_sides(source)?;
        let rhs = rhs.unwrap_or_else(|| lhs.clone());
        let symbols: std::collections::BTreeSet<_> = lhs
            .free_symbols()
            .into_iter()
            .chain(rhs.free_symbols())
            .collect();
        let bindings: Vec<_> = symbols
            .iter()
            .filter_map(|symbol| {
                self.workspace
                    .bindings
                    .get(symbol)
                    .map(|value| (symbol.clone(), value.clone()))
            })
            .collect();
        if bindings.len() > MAX_CLAIM_BINDINGS {
            return Err(SessionError::ExportLimit);
        }
        let substitution: HashMap<_, _> = bindings.iter().cloned().collect();
        let (lhs, rhs) = bounded_substitution(&lhs, &rhs, &substitution)?;
        if self.claims.len() >= self.budgets.verify_step_budget {
            return Err(SessionError::BudgetExhausted { charged: 0 });
        }
        fsym_budget::BudgetMeter::charge(&mut self.budget, Dimension::ComputeSteps, 1)
            .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
        let record = ClaimRecord {
            source: source.to_string(),
            statement: Claim::equality(lhs, rhs),
            bindings,
            status: ClaimStatus::Candidate,
            derivation_digest: None,
            context_digest: self.context_digest(),
        };
        checked_statement(&record)?;
        self.claims.push(record.clone());
        Ok(record)
    }

    /// Independently verifies a candidate claim:
    /// - For single-expression claims (`expr`), proves reflexivity + symmetry.
    /// - For equational claims (`lhs = rhs`), uses the frozen candidate bindings,
    ///   proves the equational identity via `ProofKernel` (`Reflexivity` or
    ///   `CertificatePayload::PolynomialNormalForm` + symmetry round-trip),
    ///   rejects false claims (`lhs != rhs`) with `SessionError::CandidateNotAccepted`,
    ///   and runs `verify_derivation_independent` before promoting to `Accepted`.
    pub fn verify_claim(&mut self, claim: &ClaimRecord) -> Result<ClaimRecord, SessionError> {
        if claim.context_digest != self.context_digest() {
            return Err(SessionError::UniverseMismatch {
                found: claim.context_digest,
                expected: self.context_digest(),
            });
        }
        if claim.status != ClaimStatus::Candidate || claim.derivation_digest.is_some() {
            return Err(SessionError::CandidateNotAccepted);
        }
        let Claim::Equality { lhs, rhs } = checked_statement(claim)?.clone() else {
            return Err(SessionError::ClaimMismatch);
        };
        if self.derivations.len() >= self.budgets.verify_step_budget {
            return Err(SessionError::BudgetExhausted { charged: 0 });
        }
        let meter = &mut self.budget;
        let root_step = if lhs == rhs {
            let step = self
                .kernel
                .prove_reflexivity(lhs, meter)
                .map_err(|_| SessionError::BudgetExhausted { charged: 0 })?;
            // Second step: symmetry of reflexivity gives exports a real
            // multi-step derivation so pagination is exercised honestly.
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
        };
        let derivation = self
            .kernel
            .export_derivation(root_step)
            .map_err(|_| SessionError::UnknownDerivation)?;
        // Independent re-verification: the generator (kernel) output is
        // checked by the separate verifier lane before acceptance.
        let context = ImmutableAssumptionsSnapshot::clone(&self.context);
        self.charge_verification(&derivation)?;
        let conclusion = fsym_proof_kernel::verify_derivation_independent(&derivation, &context)
            .map_err(|_| SessionError::CandidateNotAccepted)?;
        if conclusion != claim.statement {
            return Err(SessionError::ClaimMismatch);
        }
        let accepted = ClaimRecord {
            source: claim.source.clone(),
            statement: claim.statement.clone(),
            bindings: claim.bindings.clone(),
            status: ClaimStatus::Accepted,
            derivation_digest: Some(derivation.digest()),
            context_digest: claim.context_digest,
        };
        if !self.workspace.derivations.contains(&derivation) {
            self.workspace.derivations.push(derivation.clone());
            self.sync_active_branch();
        }
        self.derivations.push(derivation);
        if let Some(existing) = self.claims.iter_mut().find(|c| &**c == claim) {
            *existing = accepted.clone();
        } else {
            self.claims.push(accepted.clone());
        }
        Ok(accepted)
    }

    fn charge_verification(&mut self, derivation: &DerivationTree) -> Result<(), SessionError> {
        if derivation.steps.is_empty() || derivation.steps.len() > self.budgets.verify_step_budget {
            return Err(SessionError::BudgetExhausted { charged: 0 });
        }
        let units = fsym_proof_kernel::derivation_verification_units(derivation)
            .map_err(|_| SessionError::ExportLimit)?;
        let lease = self
            .verifier_lease
            .as_ref()
            .ok_or(SessionError::BudgetExhausted { charged: 0 })?;
        self.budget
            .try_charge_verifier(lease, units)
            .map_err(|_| SessionError::BudgetExhausted { charged: units })?;
        Ok(())
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
        if !self.claims.contains(claim) {
            return Err(SessionError::ClaimMismatch);
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
        let export = ClaimExportPage {
            schema_version: CLAIM_EXPORT_SCHEMA,
            claim: claim.clone(),
            root: derivation.root,
            page,
            total_pages,
            steps: derivation.steps[start..end].to_vec(),
        };
        bounded_serialized_size(&export)?;
        serde_json::to_value(export).map_err(|error| SessionError::Serialization(error.to_string()))
    }

    /// Replays a complete export (one page object or an ordered array of pages).
    /// No resident proof is consulted. The caller's expected context must also
    /// equal this receiving session's actual immutable context.
    pub fn replay_claim(
        &mut self,
        export: &serde_json::Value,
        context_digest: [u8; 32],
    ) -> Result<[u8; 32], SessionError> {
        if context_digest != self.context_digest() {
            return Err(SessionError::UniverseMismatch {
                found: context_digest,
                expected: self.context_digest(),
            });
        }
        let pages: Vec<ClaimExportPage> = if export.is_array() {
            decode_claim_json(export)?
        } else {
            vec![decode_claim_json(export)?]
        };
        self.replay_claim_pages(&pages)
    }

    /// Reconstructs the supplied canonical page set and checks its exact root
    /// statement using protected verifier budget. Successful replay does not
    /// import untrusted bindings into the workspace or trust an accepted flag.
    pub fn replay_claim_pages(
        &mut self,
        pages: &[ClaimExportPage],
    ) -> Result<[u8; 32], SessionError> {
        let first = pages.first().ok_or(SessionError::InvalidPages)?;
        if pages.len() > self.budgets.verify_step_budget
            || first.total_pages as usize != pages.len()
            || first.schema_version != CLAIM_EXPORT_SCHEMA
        {
            return Err(SessionError::InvalidPages);
        }
        if first.claim.context_digest != self.context_digest() {
            return Err(SessionError::UniverseMismatch {
                found: first.claim.context_digest,
                expected: self.context_digest(),
            });
        }
        if first.claim.status != ClaimStatus::Accepted {
            return Err(SessionError::CandidateNotAccepted);
        }
        checked_statement(&first.claim)?;
        let digest = first
            .claim
            .derivation_digest
            .ok_or(SessionError::UnknownDerivation)?;
        let mut count = 0usize;
        for (index, page) in pages.iter().enumerate() {
            preflight_claim_record(&page.claim)?;
            if page.schema_version != CLAIM_EXPORT_SCHEMA
                || page.page as usize != index
                || page.total_pages != first.total_pages
                || page.root != first.root
                || page.claim != first.claim
                || page.steps.is_empty()
                || page.steps.len() > MAX_EXPORT_PAGE_SIZE
                || (index + 1 < pages.len() && page.steps.len() != first.steps.len())
                || page.steps.len() > first.steps.len()
            {
                return Err(SessionError::InvalidPages);
            }
            count = count
                .checked_add(page.steps.len())
                .ok_or(SessionError::ExportLimit)?;
            if count > self.budgets.verify_step_budget {
                return Err(SessionError::BudgetExhausted { charged: 0 });
            }
            // Validate expressions before cloning or serializing their payloads.
            fsym_proof_kernel::derivation_steps_verification_units(&page.steps)
                .map_err(|_| SessionError::ExportLimit)?;
        }
        bounded_serialized_size(&pages)?;
        let derivation = DerivationTree {
            steps: pages
                .iter()
                .flat_map(|page| page.steps.iter().cloned())
                .collect(),
            root: first.root,
        };
        self.charge_verification(&derivation)?;
        let context = ImmutableAssumptionsSnapshot::clone(&self.context);
        let conclusion = fsym_proof_kernel::verify_derivation_independent(&derivation, &context)
            .map_err(|_| SessionError::ReplayDigestMismatch)?;
        if conclusion != first.claim.statement {
            return Err(SessionError::ClaimMismatch);
        }
        if derivation.digest() != digest {
            return Err(SessionError::ReplayDigestMismatch);
        }
        Ok(digest)
    }

    fn handle_claim_request(
        &mut self,
        request: ClaimRequest,
    ) -> Result<serde_json::Value, SessionError> {
        match request {
            ClaimRequest::ConstructClaim { source } => {
                serde_json::to_value(self.construct_claim(&source)?)
                    .map_err(|e| SessionError::Serialization(e.to_string()))
            }
            ClaimRequest::VerifyClaim { claim } => serde_json::to_value(self.verify_claim(&claim)?)
                .map_err(|e| SessionError::Serialization(e.to_string())),
            ClaimRequest::ExportClaim {
                claim,
                page,
                page_size,
            } => self.export_claim(&claim, page, page_size),
            ClaimRequest::ReplayClaim { pages } => {
                let digest = self.replay_claim_pages(&pages)?;
                Ok(serde_json::json!({
                    "claim": pages[0].claim,
                    "derivation_digest": digest,
                }))
            }
        }
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
        if matches!(
            value.get("type").and_then(serde_json::Value::as_str),
            Some("ConstructClaim" | "VerifyClaim" | "ExportClaim" | "ReplayClaim")
        ) {
            let result = decode_claim_json::<ClaimRequest>(&value)
                .and_then(|request| self.handle_claim_request(request));
            let mut response = match result {
                Ok(result) => serde_json::json!({ "status": "success", "result": result }),
                Err(error) => {
                    let code = match &error {
                        SessionError::Serialization(_) => "malformed_request",
                        SessionError::UniverseMismatch { .. } => "universe_mismatch",
                        SessionError::BudgetExhausted { .. } => "budget_exhausted",
                        SessionError::ExportLimit => "resource_limit_exceeded",
                        SessionError::InvalidPages => "invalid_pages",
                        SessionError::ReplayDigestMismatch | SessionError::ClaimMismatch => {
                            "invalid_proof"
                        }
                        _ => "refused",
                    };
                    serde_json::json!({ "status": "error", "code": code, "error": error.to_string() })
                }
            };
            if let Some(id) = id {
                response["id"] = id;
            }
            return response.to_string();
        }
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
