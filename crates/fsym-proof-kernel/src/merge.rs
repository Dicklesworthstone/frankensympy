//! Independent semantic merge certificates (WS14).
//!
//! The runtime constructs [`SemanticMergeCertificate`] values from an actual
//! workspace transaction; this module is the *only* authority for the
//! canonical commitment rule and hosts the minimal independent verifier that
//! rejects forged certificates. The verifier never trusts a recorded
//! `result_root`: it recomputes the commitment from the base root, the
//! sorted witnesses, the bound derivation digests, and the declared
//! policy/version inputs.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

/// Domain separator for the canonical merge commitment.
pub const MERGE_COMMITMENT_DOMAIN: &[u8] = b"fsym.proof_kernel.merge_commitment.v1\0";
/// Domain separator for workspace state roots.
pub const WORKSPACE_STATE_ROOT_DOMAIN: &[u8] = b"fsym.proof_kernel.workspace_state_root.v1\0";

/// Supported merge policies. The conservative policy refuses any overlapping
/// binding whose base and incoming values are not exactly equal; there is no
/// typed equality certificate that could justify a weaker rule yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergePolicy {
    Conservative,
}

/// The kind of observation a witness records about one symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WitnessKind {
    /// The base branch exposed a binding for the symbol; `value_digest` is
    /// the content digest of the observed value.
    Read,
    /// The merge installed a binding for the symbol; `value_digest` is the
    /// content digest of the installed value.
    Write,
    /// The base branch had no binding for the symbol; `value_digest` must be
    /// the zero digest.
    Absence,
    /// The assumption context observed during the merge; `value_digest` is
    /// the context digest.
    Predicate,
}

/// One typed observation from a merge transaction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergeWitness {
    pub symbol_digest: [u8; 32],
    pub kind: WitnessKind,
    pub value_digest: [u8; 32],
}

/// Versioned, canonical certificate for one semantic merge transaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticMergeCertificate {
    /// Certificate schema version; the verifier refuses unknown versions.
    pub policy_version: u32,
    pub policy: MergePolicy,
    /// Digest of the assumption context shared by both branches.
    pub context_digest: [u8; 32],
    /// Numeric evaluation-profile version bound by the caller.
    pub profile_version: u32,
    /// Registry (rule/claim) version bound by the caller.
    pub registry_version: u32,
    /// Content digest of the target branch state before the merge.
    pub base_root: [u8; 32],
    /// Recomputable commitment over everything above; the verifier
    /// recalculates this and refuses any mismatch.
    pub result_root: [u8; 32],
    /// Sorted witness set; the verifier refuses unsorted or duplicated
    /// entries.
    pub witnesses: Vec<MergeWitness>,
    /// Digests of every derivation imported by the merge, sorted.
    pub derivation_digests: Vec<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MergeCertError {
    #[error("merge certificate policy version {found} is not supported (expected 1)")]
    UnsupportedPolicyVersion { found: u32 },
    #[error("merge witnesses are not in canonical sorted order")]
    NonCanonicalWitnessOrder,
    #[error("witness value digest must be the zero digest for {kind:?} witnesses")]
    NonZeroDigestForWitnessKind { kind: WitnessKind },
    #[error(
        "witness write for symbol {symbol_digest:02x?} has no matching read or absence witness"
    )]
    WriteWithoutBaseObservation { symbol_digest: [u8; 32] },
    #[error("conservative policy forbids read and absence witnesses on the same symbol {0:?}")]
    ReadAbsenceOverlap([u8; 32]),
    #[error(
        "merge certificate commitment mismatch: the recorded result root does not follow from the bound inputs"
    )]
    ForgedCommitment,
    #[error("derivation digests are not in canonical sorted order")]
    NonCanonicalDerivationOrder,
}

/// The zero digest used as the canonical empty value.
pub const ZERO_DIGEST: [u8; 32] = [0u8; 32];

/// Canonical content digest of one workspace binding value.
///
/// The runtime serializes the binding expression with `serde_json` into this
/// domain; the verifier treats digests as opaque and only enforces the
/// structural rules, so no serialization logic is duplicated here.
pub fn symbol_digest(symbol_name: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.proof_kernel.merge_symbol.v1\0");
    hasher.update(symbol_name.as_bytes());
    *hasher.finalize().as_bytes()
}

/// Canonical content root of a whole workspace state (branch, assumptions,
/// sorted bindings, sorted derivation digests).
///
/// Both branches of a transaction use this for the base root; the runtime
/// computes binding values' digests with its own canonical expression
/// serialization and passes them in, keeping expression serialization in
/// exactly one crate.
pub fn workspace_state_root(
    branch_name: &str,
    assumption_digest: [u8; 32],
    bindings: &[([u8; 32], [u8; 32])], // (symbol_digest, value_digest), sorted
    derivation_digests: &[[u8; 32]],   // sorted
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(WORKSPACE_STATE_ROOT_DOMAIN);
    hasher.update(&(branch_name.len() as u64).to_le_bytes());
    hasher.update(branch_name.as_bytes());
    hasher.update(&assumption_digest);
    hasher.update(&(bindings.len() as u64).to_le_bytes());
    for (symbol, value) in bindings {
        hasher.update(symbol);
        hasher.update(value);
    }
    hasher.update(&(derivation_digests.len() as u64).to_le_bytes());
    for digest in derivation_digests {
        hasher.update(digest);
    }
    *hasher.finalize().as_bytes()
}

/// The canonical commitment rule: the single authority for how a
/// `result_root` follows from the certificate's bound inputs.
pub fn merge_commitment(
    base_root: [u8; 32],
    context_digest: [u8; 32],
    profile_version: u32,
    registry_version: u32,
    policy: MergePolicy,
    witnesses: &BTreeSet<MergeWitness>,
    derivation_digests: &BTreeSet<[u8; 32]>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MERGE_COMMITMENT_DOMAIN);
    hasher.update(&1u32.to_le_bytes()); // commitment schema version
    hasher.update(&base_root);
    hasher.update(&context_digest);
    hasher.update(&profile_version.to_le_bytes());
    hasher.update(&registry_version.to_le_bytes());
    let policy_json = serde_json::to_vec(&policy).unwrap_or_default();
    hasher.update(&(policy_json.len() as u64).to_le_bytes());
    hasher.update(&policy_json);
    hasher.update(&(witnesses.len() as u64).to_le_bytes());
    for witness in witnesses {
        let witness_json = serde_json::to_vec(witness).unwrap_or_default();
        hasher.update(&(witness_json.len() as u64).to_le_bytes());
        hasher.update(&witness_json);
    }
    hasher.update(&(derivation_digests.len() as u64).to_le_bytes());
    for digest in derivation_digests {
        hasher.update(digest);
    }
    *hasher.finalize().as_bytes()
}

/// Independent minimal verifier: re-derives the commitment from the bound
/// inputs and enforces the witness structural rules. Forged certificates —
/// altered roots, altered witnesses, unsorted sets, unobserved writes — are
/// rejected with typed errors.
pub fn verify_merge_certificate(
    certificate: &SemanticMergeCertificate,
) -> Result<(), MergeCertError> {
    if certificate.policy_version != 1 {
        return Err(MergeCertError::UnsupportedPolicyVersion {
            found: certificate.policy_version,
        });
    }

    // Canonical order: witnesses must arrive sorted (the canonical rule uses
    // BTreeSet order), without duplicates.
    for pair in certificate.witnesses.windows(2) {
        if pair[0] >= pair[1] {
            return Err(MergeCertError::NonCanonicalWitnessOrder);
        }
    }
    for witness in &certificate.witnesses {
        match witness.kind {
            WitnessKind::Absence | WitnessKind::Predicate => {
                if witness.value_digest != ZERO_DIGEST && witness.kind == WitnessKind::Absence {
                    return Err(MergeCertError::NonZeroDigestForWitnessKind {
                        kind: WitnessKind::Absence,
                    });
                }
            }
            WitnessKind::Read | WitnessKind::Write => {}
        }
    }

    // Structural rules per symbol.
    let mut symbols: BTreeSet<[u8; 32]> = BTreeSet::new();
    for witness in &certificate.witnesses {
        symbols.insert(witness.symbol_digest);
    }
    for symbol in &symbols {
        let has_read = certificate
            .witnesses
            .iter()
            .any(|w| w.symbol_digest == *symbol && w.kind == WitnessKind::Read);
        let has_absence = certificate
            .witnesses
            .iter()
            .any(|w| w.symbol_digest == *symbol && w.kind == WitnessKind::Absence);
        let has_write = certificate
            .witnesses
            .iter()
            .any(|w| w.symbol_digest == *symbol && w.kind == WitnessKind::Write);
        if has_read && has_absence {
            return Err(MergeCertError::ReadAbsenceOverlap(*symbol));
        }
        if has_write && !has_read && !has_absence {
            return Err(MergeCertError::WriteWithoutBaseObservation {
                symbol_digest: *symbol,
            });
        }
    }

    // Canonical derivation digest order.
    for pair in certificate.derivation_digests.windows(2) {
        if pair[0] >= pair[1] {
            return Err(MergeCertError::NonCanonicalDerivationOrder);
        }
    }

    // Recompute the commitment; never trust the recorded root.
    let witnesses: BTreeSet<MergeWitness> = certificate.witnesses.iter().cloned().collect();
    let derivations: BTreeSet<[u8; 32]> = certificate.derivation_digests.iter().cloned().collect();
    let recomputed = merge_commitment(
        certificate.base_root,
        certificate.context_digest,
        certificate.profile_version,
        certificate.registry_version,
        certificate.policy,
        &witnesses,
        &derivations,
    );
    if recomputed != certificate.result_root {
        return Err(MergeCertError::ForgedCommitment);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_witness(kind: WitnessKind, seed: u8) -> MergeWitness {
        let mut symbol = [0u8; 32];
        symbol[0] = seed;
        MergeWitness {
            symbol_digest: symbol,
            kind,
            value_digest: if kind == WitnessKind::Absence || kind == WitnessKind::Predicate {
                ZERO_DIGEST
            } else {
                let mut value = [0u8; 32];
                value[0] = seed.wrapping_add(100);
                value
            },
        }
    }

    fn sample_certificate() -> SemanticMergeCertificate {
        let mut witnesses = BTreeSet::new();
        witnesses.insert(sample_witness(WitnessKind::Read, 1));
        witnesses.insert(sample_witness(WitnessKind::Write, 1));
        witnesses.insert(sample_witness(WitnessKind::Absence, 2));
        witnesses.insert(sample_witness(WitnessKind::Predicate, 3));
        let mut derivations = BTreeSet::new();
        derivations.insert([7u8; 32]);
        let base_root = workspace_state_root("base", [1u8; 32], &[], &[]);
        let result_root = merge_commitment(
            base_root,
            [1u8; 32],
            1,
            1,
            MergePolicy::Conservative,
            &witnesses,
            &derivations,
        );
        SemanticMergeCertificate {
            policy_version: 1,
            policy: MergePolicy::Conservative,
            context_digest: [1u8; 32],
            profile_version: 1,
            registry_version: 1,
            base_root,
            result_root,
            witnesses: witnesses.into_iter().collect(),
            derivation_digests: derivations.into_iter().collect(),
        }
    }

    #[test]
    fn well_formed_certificate_verifies() {
        assert!(verify_merge_certificate(&sample_certificate()).is_ok());
    }

    #[test]
    fn forged_result_root_is_rejected() {
        let mut cert = sample_certificate();
        cert.result_root[0] ^= 0xFF;
        assert_eq!(
            verify_merge_certificate(&cert),
            Err(MergeCertError::ForgedCommitment)
        );
    }

    #[test]
    fn altered_witness_is_rejected_as_forgery() {
        let mut cert = sample_certificate();
        cert.witnesses[0].value_digest[0] ^= 0xFF;
        assert_eq!(
            verify_merge_certificate(&cert),
            Err(MergeCertError::ForgedCommitment)
        );
    }

    #[test]
    fn unsorted_witnesses_are_rejected() {
        let mut cert = sample_certificate();
        cert.witnesses.reverse();
        assert_eq!(
            verify_merge_certificate(&cert),
            Err(MergeCertError::NonCanonicalWitnessOrder)
        );
    }

    #[test]
    fn read_absence_overlap_is_refused_conservatively() {
        let mut cert = sample_certificate();
        cert.witnesses.push(sample_witness(WitnessKind::Absence, 1));
        cert.witnesses.sort();
        // Recompute a matching root so only the structural rule can fire.
        let witnesses: BTreeSet<MergeWitness> = cert.witnesses.iter().cloned().collect();
        cert.result_root = merge_commitment(
            cert.base_root,
            cert.context_digest,
            cert.profile_version,
            cert.registry_version,
            cert.policy,
            &witnesses,
            &BTreeSet::from([[7u8; 32]]),
        );
        let symbol = symbol_for_seed(1);
        assert_eq!(
            verify_merge_certificate(&cert),
            Err(MergeCertError::ReadAbsenceOverlap(symbol))
        );
    }

    #[test]
    fn write_without_base_observation_is_rejected() {
        let mut cert = sample_certificate();
        cert.witnesses
            .retain(|w| !(w.symbol_digest[0] == 1 && (w.kind == WitnessKind::Read)));
        cert.witnesses.sort();
        let witnesses: BTreeSet<MergeWitness> = cert.witnesses.iter().cloned().collect();
        cert.result_root = merge_commitment(
            cert.base_root,
            cert.context_digest,
            cert.profile_version,
            cert.registry_version,
            cert.policy,
            &witnesses,
            &BTreeSet::from([[7u8; 32]]),
        );
        assert_eq!(
            verify_merge_certificate(&cert),
            Err(MergeCertError::WriteWithoutBaseObservation {
                symbol_digest: symbol_for_seed(1),
            })
        );
    }

    #[test]
    fn unsupported_policy_version_is_rejected() {
        let mut cert = sample_certificate();
        cert.policy_version = 2;
        assert_eq!(
            verify_merge_certificate(&cert),
            Err(MergeCertError::UnsupportedPolicyVersion { found: 2 })
        );
    }

    fn symbol_for_seed(seed: u8) -> [u8; 32] {
        let mut symbol = [0u8; 32];
        symbol[0] = seed;
        symbol
    }
}
