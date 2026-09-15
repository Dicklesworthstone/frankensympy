//! Publication gating for verified workspace merges (WS14).
//!
//! A [`PublicationGate`] owns the lease epoch and the poison state of one
//! publication channel. Only a merge whose certificate carries the *current*
//! lease epoch, an unpoisoned channel, and a `verified = true` marker may
//! publish. Every refusal is typed so callers cannot confuse a stale lease
//! with a poisoned channel or an unverified candidate.

#![forbid(unsafe_code)]

use fsym_proof_kernel::SemanticMergeCertificate;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PublicationError {
    #[error("publication refused: the lease epoch {provided} is stale (current is {current})")]
    StaleLeaseEpoch { current: u64, provided: u64 },
    #[error("publication refused: the channel is poisoned: {0}")]
    ChannelPoisoned(String),
    #[error("publication refused: the candidate was never independently verified")]
    UnverifiedCandidate,
    #[error("publication refused: {0}")]
    Refused(String),
}

/// One publication channel: at most one live lease, one published root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicationGate {
    lease_epoch: u64,
    poison_reason: Option<String>,
    published_root: Option<[u8; 32]>,
    history: Vec<[u8; 32]>,
}

impl PublicationGate {
    pub fn new() -> Self {
        Self {
            lease_epoch: 1,
            poison_reason: None,
            published_root: None,
            history: Vec::new(),
        }
    }

    /// The currently live lease epoch. A caller holding an older epoch must
    /// renew before publishing.
    pub fn lease_epoch(&self) -> u64 {
        self.lease_epoch
    }

    /// Renew the lease: invalidates every older epoch.
    pub fn renew_lease(&mut self) -> u64 {
        self.lease_epoch = self.lease_epoch.saturating_add(1);
        self.lease_epoch
    }

    /// Poison the channel. A poisoned channel refuses every publication
    /// until a caller explicitly restores it (a deliberate, separate step —
    /// recovery is never implicit).
    pub fn poison(&mut self, reason: impl Into<String>) {
        self.poison_reason = Some(reason.into());
    }

    /// Explicitly clear a poison marker.
    pub fn restore(&mut self) {
        self.poison_reason = None;
    }

    pub fn poison_reason(&self) -> Option<&str> {
        self.poison_reason.as_deref()
    }

    pub fn published_root(&self) -> Option<[u8; 32]> {
        self.published_root
    }

    /// Retention history of every published root, oldest first. Replay and
    /// audit paths that need prior roots must consult this; a missing
    /// history entry is a typed refusal, never a silent fallback.
    pub fn retention_history(&self) -> &[[u8; 32]] {
        &self.history
    }

    /// Publish a merge certificate. Refuses stale lease epochs, poisoned
    /// channels, and candidates that were never independently verified.
    pub fn publish(
        &mut self,
        lease_epoch: u64,
        certificate: &SemanticMergeCertificate,
        verified: bool,
    ) -> Result<[u8; 32], PublicationError> {
        if let Some(reason) = &self.poison_reason {
            return Err(PublicationError::ChannelPoisoned(reason.clone()));
        }
        if lease_epoch != self.lease_epoch {
            return Err(PublicationError::StaleLeaseEpoch {
                current: self.lease_epoch,
                provided: lease_epoch,
            });
        }
        if !verified {
            return Err(PublicationError::UnverifiedCandidate);
        }
        self.published_root = Some(certificate.result_root);
        self.history.push(certificate.result_root);
        Ok(certificate.result_root)
    }

    /// Look up a previously published root; missing retention history is a
    /// typed refusal, never a silent fallback.
    pub fn lookup_history(&self, position: usize) -> Result<[u8; 32], PublicationError> {
        self.history.get(position).copied().ok_or_else(|| {
            PublicationError::Refused(format!(
                "retention history has no entry at position {position} \
                     (published count: {})",
                self.history.len()
            ))
        })
    }
}

impl Default for PublicationGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_proof_kernel::{MergePolicy, MergeWitness, WitnessKind};

    fn sample_certificate() -> SemanticMergeCertificate {
        let mut witness = MergeWitness {
            symbol_digest: [1u8; 32],
            kind: WitnessKind::Absence,
            value_digest: [0u8; 32],
        };
        witness.symbol_digest[1] = 7;
        let base_root = [3u8; 32];
        let mut witnesses = std::collections::BTreeSet::new();
        witnesses.insert(witness);
        let derivations = std::collections::BTreeSet::new();
        let result_root = fsym_proof_kernel::merge_commitment(
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
            derivation_digests: Vec::new(),
        }
    }

    #[test]
    fn stale_lease_epoch_cannot_publish() {
        let mut gate = PublicationGate::new();
        let cert = sample_certificate();
        assert_eq!(
            gate.publish(0, &cert, true),
            Err(PublicationError::StaleLeaseEpoch {
                current: 1,
                provided: 0
            })
        );
    }

    #[test]
    fn renewed_lease_invalidates_the_old_epoch() {
        let mut gate = PublicationGate::new();
        let old = gate.lease_epoch();
        gate.renew_lease();
        let cert = sample_certificate();
        assert!(matches!(
            gate.publish(old, &cert, true),
            Err(PublicationError::StaleLeaseEpoch { .. })
        ));
        assert!(gate.publish(gate.lease_epoch(), &cert, true).is_ok());
    }

    #[test]
    fn poisoned_channel_refuses_until_restored() {
        let mut gate = PublicationGate::new();
        gate.poison("quarantined by monitor");
        let cert = sample_certificate();
        assert_eq!(
            gate.publish(gate.lease_epoch(), &cert, true),
            Err(PublicationError::ChannelPoisoned(
                "quarantined by monitor".to_string()
            ))
        );
        gate.restore();
        assert!(gate.publish(gate.lease_epoch(), &cert, true).is_ok());
    }

    #[test]
    fn unverified_candidate_cannot_publish() {
        let mut gate = PublicationGate::new();
        let cert = sample_certificate();
        assert_eq!(
            gate.publish(gate.lease_epoch(), &cert, false),
            Err(PublicationError::UnverifiedCandidate)
        );
        assert!(gate.publish(gate.lease_epoch(), &cert, true).is_ok());
    }

    #[test]
    fn missing_history_entry_is_a_typed_refusal() {
        let gate = PublicationGate::new();
        assert!(matches!(
            gate.lookup_history(0),
            Err(PublicationError::Refused(_))
        ));
        let mut gate = PublicationGate::new();
        let cert = sample_certificate();
        let _ = gate.publish(gate.lease_epoch(), &cert, true);
        assert_eq!(gate.lookup_history(0), Ok(cert.result_root));
        assert!(gate.lookup_history(1).is_err());
    }
}
