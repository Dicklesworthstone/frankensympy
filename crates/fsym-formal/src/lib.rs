//! Optional native-first formal projection (WS06, bead `fra-rc-formal-ma3`).
//!
//! Translates verifier-complete native capsule claims into pinned Lean 4
//! statements and issues projection receipts binding the native claim root to
//! the formal statement root. Native verification never depends on this crate:
//! `verify_capsule` in `fsym-proof-kernel` remains the sole native authority,
//! and every refusal here leaves the native evidence class untouched.

#![forbid(unsafe_code)]

use fsym_proof_kernel::capsule::{Capsule, PolyObject};
use serde::{Deserialize, Serialize};

mod admission;
pub mod checker;
mod envelope;
pub use envelope::{CheckedProjection, ProjectionEnvelope, decode_hex, encode_hex};

pub const MAX_CAPSULE_BYTES: usize = 16 * 1024;
pub const MAX_SOURCE_BYTES: usize = 32 * 1024;
pub const MAX_JSON_BYTES: usize = 128 * 1024;
pub const MAX_DEGREE: u32 = 64;
pub const CONTEXT_ROOT: u64 = 0x1111222233334444;
pub const RULE_ROOT: u64 = 0x5555666677778888;
pub const VERIFIER_ROOT: u64 = 0x9999aaaabbbbcccc;

/// Cooperative cancellation checkpoints; true means cancel, never reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafePoint {
    Admission,
    PreflightObject,
    BeforeNativeVerification,
    AfterNativeVerification,
    BeforeOutput,
}

pub(crate) fn checkpoint(
    cancel: &impl Fn(SafePoint) -> bool,
    point: SafePoint,
) -> Result<(), ProjectionError> {
    if cancel(point) {
        Err(ProjectionError::Cancelled)
    } else {
        Ok(())
    }
}

/// Projection profile implemented by this adapter. Frozen: any mapping change
/// mints a new profile ID, never an edit of this one.
pub const PROFILE_ID: &str = "lean_core_zz_product_v1";
/// Pinned foreign checker toolchain (exact `lean --version` output).
pub const CHECKER_PIN: &str = "Lean (version 4.32.2, x86_64-unknown-linux-gnu, commit f3b06c705e6c85f5314019d5d3baab0fec5b580c, Release)";
/// SHA256 of the pinned checker executable and installed library closure.
pub const CHECKER_ENVIRONMENT_SHA256: &str =
    "88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f";
/// Exact environment identifier bound into both statement and receipt roots.
pub const STATEMENT_ENVIRONMENT_ROOT: &str = "lean4-core-4.32.2@f3b06c705e6c85f5314019d5d3baab0fec5b580c;sha256=88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProjectionError {
    #[error("profile {PROFILE_ID} projects only ZZ factor-product claims")]
    UnsupportedFamily,
    #[error("capsule shape does not match the frozen subject/scalar/factor mapping")]
    UnsupportedShape,
    #[error("semantic field outside the profile mapping (never ignored)")]
    UnrepresentableSemanticField,
    #[error("native verification failed")]
    NativeVerificationFailed,
    #[error("malformed or noncanonical input")]
    MalformedInput,
    #[error("projection receipt or statement mismatch")]
    ProjectionReceiptMismatch,
    #[error("projection resource exhausted: {0}")]
    ResourceExhausted(&'static str),
    #[error("projection cancelled")]
    Cancelled,
}

/// Untrusted serialized binding. Only `checker::check_projection` mints checked state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionReceipt {
    pub profile_id: String,
    pub checker_pin: String,
    pub environment: String,
    pub native_claim_root: String,
    pub capsule_root: String,
    pub formal_statement_root: String,
    pub schema_version: u16,
    pub domain_root: u64,
    pub context_root: u64,
    pub rule_root: u64,
    pub verifier_root: u64,
    pub semantic_fields: Vec<String>,
    pub receipt_root: String,
}

fn statement_root(environment: &str, profile: &str, source: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.formal.statement.v1");
    hasher.update(environment.as_bytes());
    hasher.update(&[0]);
    hasher.update(profile.as_bytes());
    hasher.update(&[0]);
    hasher.update(source.as_bytes());
    *hasher.finalize().as_bytes()
}

fn make_receipt(
    bytes: &[u8],
    capsule: &Capsule,
    native_claim_root: &[u8; 32],
    formal_statement_root: &[u8; 32],
) -> ProjectionReceipt {
    let mut receipt = ProjectionReceipt {
        profile_id: PROFILE_ID.into(),
        checker_pin: CHECKER_PIN.into(),
        environment: STATEMENT_ENVIRONMENT_ROOT.into(),
        native_claim_root: encode_hex(native_claim_root),
        capsule_root: encode_hex(blake3::hash(bytes).as_bytes()),
        formal_statement_root: encode_hex(formal_statement_root),
        schema_version: capsule.schema_version,
        domain_root: capsule.claim.domain.root().raw(),
        context_root: capsule.claim.context.raw(),
        rule_root: capsule.claim.rule.raw(),
        verifier_root: capsule.claim.verifier.raw(),
        semantic_fields: [
            "domain=ZZ",
            "context=no-hypotheses",
            "scalar=1",
            "factors=2;exponents=1",
            "equality=exact-coefficient-convolution",
            "binding=one-common-native-symbol",
            "branches=none;partiality=none;extensions=none",
            "schema=fsym.capsule.poly-identity.v1",
            "evidence=projection-mapping-only;foreign=not-checked",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        receipt_root: String::new(),
    };
    // Length-delimited canonical struct serialization binds every receipt field.
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.formal.zz-product.receipt.v1\0");
    hasher.update(
        &serde_json::to_vec(&receipt).expect("receipt contains only serializable primitives"),
    );
    receipt.receipt_root = encode_hex(hasher.finalize().as_bytes());
    receipt
}

/// Dense ascending coefficient vector of a capsule polynomial.
///
/// The capsule stores descending sparse `(degree, coefficient)` terms; the
/// Lean statement consumes the dense ascending vector. Non-integral or
/// out-of-`i64` coefficients refuse rather than approximate.
pub fn dense_ascending(poly: &PolyObject) -> Result<Vec<i64>, ProjectionError> {
    if poly.terms.is_empty() {
        return Err(ProjectionError::UnsupportedShape);
    }
    if poly.terms.len() > 65 || poly.terms.iter().any(|(degree, _)| *degree > MAX_DEGREE) {
        return Err(ProjectionError::ResourceExhausted("degree/terms"));
    }
    if poly.terms.windows(2).any(|terms| terms[0].0 <= terms[1].0) {
        return Err(ProjectionError::MalformedInput);
    }
    for (_, coefficient) in &poly.terms {
        if !coefficient.is_integer() {
            return Err(ProjectionError::UnrepresentableSemanticField);
        }
        if coefficient.numer().to_i64().is_none() {
            return Err(ProjectionError::ResourceExhausted("coefficient height"));
        }
    }
    let mut coeffs = vec![0i64; poly.degree() as usize + 1];
    for &(degree, ref coefficient) in &poly.terms {
        if !coefficient.is_integer() {
            return Err(ProjectionError::UnrepresentableSemanticField);
        }
        let value: i64 = coefficient
            .numer()
            .to_i64()
            .ok_or(ProjectionError::UnrepresentableSemanticField)?;
        let slot = &mut coeffs[degree as usize];
        if *slot != 0 {
            return Err(ProjectionError::UnsupportedShape);
        }
        *slot = value;
    }
    Ok(coeffs)
}

fn render_vector(coeffs: &[i64]) -> String {
    let rendered: Vec<String> = coeffs.iter().map(|c| c.to_string()).collect();
    format!("[{}]", rendered.join(", "))
}

/// Frozen core-only definitions, shared as a language specification with the checker.
pub(crate) const LEAN_PREFIX: &str = "-- Projected statement for profile lean_core_zz_product_v1 (fixed environment: Lean 4 core only).\n\
-- Projection: dense ascending coefficient vectors; equality = coefficient-wise after\n\
-- convolution product, which is structurally recursive and reduces by `decide`.\n\n\
/-- Pad-and-add: add `scaled` into `acc` indexwise; surplus coefficients appended. -/\n\
def addPad : List Int → List Int → List Int\n\
\x20 | [], r => r\n\
\x20 | s :: stail, [] => s :: stail\n\
\x20 | s :: stail, r0 :: rtail => (s + r0) :: addPad stail rtail\n\n\
/-- Convolution product of dense ascending ZZ coefficient vectors. -/\n\
def polyMul : List Int → List Int → List Int\n\
\x20 | [], _ => []\n\
\x20 | _, [] => []\n\
\x20 | a :: atail, b => addPad (b.map (· * a)) (0 :: polyMul atail b)\n\n";

fn render_statement(subject: &[i64], factor_a: &[i64], factor_b: &[i64]) -> String {
    format!(
        "{LEAN_PREFIX}theorem projected_product :\n    polyMul {} {} = {} := by decide\n#print axioms projected_product\n",
        render_vector(factor_a),
        render_vector(factor_b),
        render_vector(subject)
    )
}

/// Verify canonical native bytes against an independently supplied root, then
/// project only the bounded coefficient identity. No foreign process is run.
pub fn project_capsule(
    bytes: &[u8],
    native_claim_root: [u8; 32],
    cancel: &impl Fn(SafePoint) -> bool,
) -> Result<CheckedProjection, ProjectionError> {
    let capsule = admission::admit(bytes, native_claim_root, cancel)?;
    let subject = capsule
        .polynomial(capsule.claim.subject)
        .map_err(|_| ProjectionError::MalformedInput)?;
    let a = capsule
        .polynomial(capsule.claim.factors[0].0)
        .map_err(|_| ProjectionError::MalformedInput)?;
    let b = capsule
        .polynomial(capsule.claim.factors[1].0)
        .map_err(|_| ProjectionError::MalformedInput)?;
    let lean_source = render_statement(
        &dense_ascending(&subject)?,
        &dense_ascending(&a)?,
        &dense_ascending(&b)?,
    );
    let root = statement_root(STATEMENT_ENVIRONMENT_ROOT, PROFILE_ID, &lean_source);
    let envelope = ProjectionEnvelope {
        profile_id: PROFILE_ID.into(),
        checker_pin: CHECKER_PIN.into(),
        environment: STATEMENT_ENVIRONMENT_ROOT.into(),
        capsule_hex: encode_hex(bytes),
        native_claim_root: encode_hex(&native_claim_root),
        statement_root: encode_hex(&root),
        lean_source,
        receipt: make_receipt(bytes, &capsule, &native_claim_root, &root),
        native_verdict: "verified".into(),
    };
    checker::check_projection(&envelope, native_claim_root, cancel)
}
