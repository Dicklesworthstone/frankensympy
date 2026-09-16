//! Canonical FMAP bundles and fresh-process workspace replay (WS14).
//!
//! An [`FmapBundle`] separates the *verifier-complete cut* — the only
//! material an independent verifier needs to re-check the merge — from the
//! *replay metadata*, which records how the transaction was produced but is
//! never trusted as evidence. Fresh-process replay reconstructs both branch
//! workspaces from the cut, re-runs the merge, re-builds the certificate,
//! and requires full equality with the recorded certificate before
//! accepting the recorded result root.

#![forbid(unsafe_code)]

use crate::workspace::SemanticWorkspace;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{DerivationTree, SemanticMergeCertificate};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bundle schema version; replay refuses anything else.
pub const FMAP_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FmapError {
    #[error("FMAP bundle schema version {found} is not supported (expected {expected})")]
    UnsupportedSchemaVersion { found: u32, expected: u32 },
    #[error(
        "FMAP bundle rejected: the embedded merge certificate failed independent verification: {0}"
    )]
    ForgedCertificate(String),
    #[error(
        "FMAP replay mismatch: recomputed result root {recomputed:02x?} does not match the expected root {expected:02x?}"
    )]
    ReplayRootMismatch {
        recomputed: [u8; 32],
        expected: [u8; 32],
    },
    #[error(
        "FMAP replay refused: the recorded certificate differs from the replayed transaction: {0}"
    )]
    CertificateDiverged(String),
    #[error("FMAP bundle serialization failed: {0}")]
    Serialization(String),
    #[error("FMAP bundle exceeds the declared size limit of {limit} bytes")]
    SizeLimit { limit: usize },
}

/// The verified content of one merge transaction: everything an independent
/// verifier is allowed to consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierCompleteCut {
    pub certificate: SemanticMergeCertificate,
    pub assumption_context: ImmutableAssumptionsSnapshot,
    pub base_branch_name: String,
    pub source_branch_name: String,
    /// Target branch bindings before the merge, canonical (sorted) order.
    pub base_bindings: Vec<(Symbol, Expr)>,
    /// Source branch bindings (full state), canonical (sorted) order.
    pub source_bindings: Vec<(Symbol, Expr)>,
    pub base_derivations: Vec<DerivationTree>,
    pub source_derivations: Vec<DerivationTree>,
}

/// How the transaction was produced. Explicitly *not* evidence: replay
/// metadata never enters the certificate commitment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayMetadata {
    pub initial_seed: u64,
    /// Trace normal form: the deterministic event names, in order.
    pub trace_normal_form: Vec<String>,
    pub profile_version: u32,
    pub registry_version: u32,
    pub expected_result_root: [u8; 32],
}

/// Canonical Fork/Merge/As-Published bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FmapBundle {
    pub schema_version: u32,
    pub verifier_complete_cut: VerifierCompleteCut,
    pub replay_metadata: ReplayMetadata,
}

/// The outcome of a successful fresh-process replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayOutcome {
    pub result_root: [u8; 32],
    pub base_root: [u8; 32],
    pub certificate_matches: bool,
}

impl FmapBundle {
    /// Captures a completed merge transaction into a canonical bundle.
    ///
    /// `base` is the target branch *before* the merge, `merged` the target
    /// branch *after* the merge, `source` the incoming branch; `certificate`
    /// is the certificate produced by the merge.
    pub fn capture(
        base: &SemanticWorkspace,
        merged: &SemanticWorkspace,
        source: &SemanticWorkspace,
        certificate: SemanticMergeCertificate,
        replay_metadata: ReplayMetadata,
    ) -> Self {
        let mut base_bindings: Vec<(Symbol, Expr)> = base
            .bindings
            .iter()
            .map(|(s, e)| (s.clone(), e.clone()))
            .collect();
        base_bindings.sort_by(|a, b| a.0.name.cmp(&b.0.name));
        let mut source_bindings: Vec<(Symbol, Expr)> = source
            .bindings
            .iter()
            .map(|(s, e)| (s.clone(), e.clone()))
            .collect();
        source_bindings.sort_by(|a, b| a.0.name.cmp(&b.0.name));
        Self {
            schema_version: FMAP_SCHEMA_VERSION,
            verifier_complete_cut: VerifierCompleteCut {
                certificate,
                assumption_context: ImmutableAssumptionsSnapshot::clone(&merged.assumptions),
                base_branch_name: base.branch_name.clone(),
                source_branch_name: source.branch_name.clone(),
                base_bindings,
                source_bindings,
                base_derivations: base.derivations.clone(),
                source_derivations: source.derivations.clone(),
            },
            replay_metadata,
        }
    }

    /// Canonical JSON encoding. The cut and the metadata are serialized as
    /// one document but remain logically separate: only the cut may feed a
    /// verifier.
    pub fn to_json(&self) -> Result<Vec<u8>, FmapError> {
        let mut bytes = serde_json::to_vec(&self)
            .map_err(|error| FmapError::Serialization(error.to_string()))?;
        if bytes.len() > MAX_FMAP_BYTES {
            return Err(FmapError::SizeLimit {
                limit: MAX_FMAP_BYTES,
            });
        }
        bytes.shrink_to_fit();
        Ok(bytes)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, FmapError> {
        if bytes.len() > MAX_FMAP_BYTES {
            return Err(FmapError::SizeLimit {
                limit: MAX_FMAP_BYTES,
            });
        }
        let bundle: Self = serde_json::from_slice(bytes)
            .map_err(|error| FmapError::Serialization(error.to_string()))?;
        if bundle.schema_version != FMAP_SCHEMA_VERSION {
            return Err(FmapError::UnsupportedSchemaVersion {
                found: bundle.schema_version,
                expected: FMAP_SCHEMA_VERSION,
            });
        }
        Ok(bundle)
    }
}

pub const MAX_FMAP_BYTES: usize = 16 * 1024 * 1024;

/// Fresh-process replay: rebuild both branches from the cut, re-run the
/// merge, and require the replayed certificate to equal the recorded one
/// byte-for-byte. The recorded root is accepted only after that equality.
pub fn replay_fmap_bundle(bundle: &FmapBundle) -> Result<ReplayOutcome, FmapError> {
    let cut = &bundle.verifier_complete_cut;
    // 1. The certificate must survive the independent kernel verifier.
    fsym_proof_kernel::verify_merge_certificate(&cut.certificate)
        .map_err(|error| FmapError::ForgedCertificate(error.to_string()))?;

    // 2. Rebuild both branches from the cut alone.
    let mut base = SemanticWorkspace::new(cut.base_branch_name.clone());
    base.assumptions = std::sync::Arc::new(cut.assumption_context.clone());
    for (symbol, expr) in &cut.base_bindings {
        base.bind(symbol.clone(), expr.clone());
    }
    base.derivations = cut.base_derivations.clone();

    let mut source = SemanticWorkspace::new(cut.source_branch_name.clone());
    source.assumptions = std::sync::Arc::new(cut.assumption_context.clone());
    for (symbol, expr) in &cut.source_bindings {
        source.bind(symbol.clone(), expr.clone());
    }
    source.derivations = cut.source_derivations.clone();

    // 3. Re-run the merge and compare certificates.
    let replayed = base
        .merge_with_certificate(
            &source,
            bundle.replay_metadata.profile_version,
            bundle.replay_metadata.registry_version,
        )
        .map_err(|error| FmapError::CertificateDiverged(error.to_string()))?;
    let certificate_matches = replayed == cut.certificate;
    if !certificate_matches {
        return Err(FmapError::CertificateDiverged(
            "replayed certificate differs from the recorded certificate".to_string(),
        ));
    }

    // 4. The recorded expected root must agree with the certificate.
    if bundle.replay_metadata.expected_result_root != cut.certificate.result_root {
        return Err(FmapError::ReplayRootMismatch {
            recomputed: cut.certificate.result_root,
            expected: bundle.replay_metadata.expected_result_root,
        });
    }

    Ok(ReplayOutcome {
        result_root: cut.certificate.result_root,
        base_root: cut.certificate.base_root,
        certificate_matches,
    })
}

#[cfg(test)]
mod tamper_fuzz_tests {
    use super::*;

    /// Deterministic boundary fuzz: flip bytes, truncate, and splice a
    /// valid bundle; every outcome must be a typed refusal or an untouched
    /// success. A panic on any mutation is a test failure.
    #[test]
    fn bundle_tamper_fuzz_never_panics_and_never_forges() {
        let base = SemanticWorkspace::new("base");
        let mut source = SemanticWorkspace::new("feature");
        source.bind(
            Symbol::new("x"),
            Expr::Sym(Symbol::new("y")),
        );
        let mut merged = base.clone();
        let cert = merged
            .merge_with_certificate(&source, 1, 1)
            .expect("merge");
        let metadata = ReplayMetadata {
            initial_seed: 9,
            trace_normal_form: vec!["fork".to_string(), "merge".to_string()],
            profile_version: 1,
            registry_version: 1,
            expected_result_root: cert.result_root,
        };
        let bundle = FmapBundle::capture(&base, &merged, &source, cert, metadata);
        let original = bundle.to_json().expect("serializes");

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut refused = 0usize;
        let mut accepted = 0usize;
        for _ in 0..20_000 {
            let mut mutated = original.clone();
            match next() % 4 {
                0 => {
                    // single-byte flip at a pseudo-random position
                    let pos = (next() as usize) % mutated.len();
                    mutated[pos] ^= (next() % 255 + 1) as u8;
                }
                1 => {
                    // truncate
                    let pos = (next() as usize) % mutated.len();
                    mutated.truncate(pos);
                }
                2 => {
                    // delete a span
                    let pos = (next() as usize) % mutated.len();
                    let len = ((next() as usize) % 64).min(mutated.len() - pos);
                    mutated.drain(pos..pos + len);
                }
                _ => {
                    // splice garbage
                    let pos = (next() as usize) % mutated.len();
                    let junk = vec![b'x'; (next() as usize) % 32];
                    mutated.splice(pos..pos, junk);
                }
            }
            match FmapBundle::from_json(&mutated) {
                Err(_) => {
                    refused += 1;
                }
                Ok(parsed) => match replay_fmap_bundle(&parsed) {
                    Err(_) => refused += 1,
                    Ok(outcome) => {
                        // Acceptance is only sound for the untouched bundle:
                        // the mutated document must be byte-identical.
                        assert_eq!(
                            parsed, bundle,
                            "a mutated bundle was accepted"
                        );
                        assert_eq!(
                            outcome.result_root,
                            bundle.replay_metadata.expected_result_root
                        );
                        accepted += 1;
                    }
                },
            }
        }
        // The untouched case must appear among the accepted outcomes.
        assert!(accepted >= 1, "the pristine bundle must still replay");
        assert!(refused > 0, "tampering must be visibly refused");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::parse;

    #[test]
    fn bundle_round_trips_through_json() {
        let base = SemanticWorkspace::new("base");
        let mut source = SemanticWorkspace::new("feature");
        let expr = parse("x + 1").expect("parses");
        source.bind(Symbol::new("x"), expr);
        let mut merged = base.clone();
        merged.branch_name = "base".to_string();
        let cert = merged.merge_with_certificate(&source, 1, 1).expect("merge");
        let metadata = ReplayMetadata {
            initial_seed: 42,
            trace_normal_form: vec!["fork".to_string(), "patch".to_string(), "merge".to_string()],
            profile_version: 1,
            registry_version: 1,
            expected_result_root: cert.result_root,
        };
        let bundle = FmapBundle::capture(&base, &merged, &source, cert, metadata);
        let json = bundle.to_json().expect("serializes");
        let decoded = FmapBundle::from_json(&json).expect("deserializes");
        assert_eq!(bundle, decoded);
        let outcome = replay_fmap_bundle(&decoded).expect("replays");
        assert!(outcome.certificate_matches);
        assert_eq!(
            outcome.result_root,
            decoded.replay_metadata.expected_result_root
        );
    }

    #[test]
    fn tampered_bundle_is_refused_by_the_kernel_verifier() {
        let base = SemanticWorkspace::new("base");
        let mut source = SemanticWorkspace::new("feature");
        source.bind(Symbol::new("x"), parse("x + 1").expect("parses"));
        let mut merged = base.clone();
        let cert = merged.merge_with_certificate(&source, 1, 1).expect("merge");
        let metadata = ReplayMetadata {
            initial_seed: 7,
            trace_normal_form: vec![],
            profile_version: 1,
            registry_version: 1,
            expected_result_root: cert.result_root,
        };
        let mut bundle = FmapBundle::capture(&base, &merged, &source, cert, metadata);
        bundle.verifier_complete_cut.certificate.result_root[0] ^= 0xFF;
        match replay_fmap_bundle(&bundle) {
            Err(FmapError::ForgedCertificate(_)) => {}
            other => panic!("expected a forged-certificate refusal, got {other:?}"),
        }
    }

    #[test]
    fn reordered_trace_metadata_cannot_change_the_roots() {
        let base = SemanticWorkspace::new("base");
        let mut source = SemanticWorkspace::new("feature");
        source.bind(Symbol::new("x"), parse("x + 1").expect("parses"));
        let mut merged = base.clone();
        let cert = merged.merge_with_certificate(&source, 1, 1).expect("merge");
        let root = cert.result_root;
        let metadata_a = ReplayMetadata {
            initial_seed: 1,
            trace_normal_form: vec!["a".to_string(), "b".to_string()],
            profile_version: 1,
            registry_version: 1,
            expected_result_root: root,
        };
        let metadata_b = ReplayMetadata {
            trace_normal_form: vec!["b".to_string(), "a".to_string()],
            initial_seed: 2,
            ..metadata_a.clone()
        };
        let bundle_a = FmapBundle::capture(&base, &merged, &source, cert.clone(), metadata_a);
        let bundle_b = FmapBundle::capture(&base, &merged, &source, cert, metadata_b);
        // The verified cut is identical; metadata ordering is not evidence.
        assert_eq!(
            bundle_a.verifier_complete_cut,
            bundle_b.verifier_complete_cut
        );
        assert_eq!(
            replay_fmap_bundle(&bundle_a).expect("replays").result_root,
            replay_fmap_bundle(&bundle_b).expect("replays").result_root
        );
    }

    #[test]
    fn wrong_expected_root_is_a_typed_mismatch() {
        let base = SemanticWorkspace::new("base");
        let mut source = SemanticWorkspace::new("feature");
        source.bind(Symbol::new("x"), parse("x + 1").expect("parses"));
        let mut merged = base.clone();
        let cert = merged.merge_with_certificate(&source, 1, 1).expect("merge");
        let metadata = ReplayMetadata {
            initial_seed: 3,
            trace_normal_form: vec![],
            profile_version: 1,
            registry_version: 1,
            expected_result_root: [9u8; 32],
        };
        let bundle = FmapBundle::capture(&base, &merged, &source, cert, metadata);
        match replay_fmap_bundle(&bundle) {
            Err(FmapError::ReplayRootMismatch { .. }) => {}
            other => panic!("expected a replay root mismatch, got {other:?}"),
        }
    }
}
