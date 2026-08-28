//! Evidence envelopes packaging claims, receipts, and derivation trees (WS06).
//!
//! Layer: L2 (evidence envelopes).

#![forbid(unsafe_code)]

use crate::receipt::VerificationReceipt;
use fsym_outcome::EvidenceClass;
use fsym_proof_kernel::{
    Claim, DerivationTree, claim_verification_units, derivation_verification_units,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Sealed evidence envelope packaging a claim and its verification credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceEnvelope {
    pub claim: Claim,
    pub evidence_class: EvidenceClass,
    pub receipt: VerificationReceipt,
    pub derivation: Option<DerivationTree>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvelopeWire {
    claim: Claim,
    evidence_class: String,
    receipt: VerificationReceipt,
    derivation: Option<DerivationTree>,
}

impl Serialize for EvidenceEnvelope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = EnvelopeWire {
            claim: self.claim.clone(),
            evidence_class: self.evidence_class.as_str().to_string(),
            receipt: self.receipt.clone(),
            derivation: self.derivation.clone(),
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EvidenceEnvelope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EnvelopeWire::deserialize(deserializer)?;
        let evidence_class = EvidenceClass::parse(&wire.evidence_class).ok_or_else(|| {
            serde::de::Error::custom(format!("unknown evidence class `{}`", wire.evidence_class))
        })?;
        Ok(Self {
            claim: wire.claim,
            evidence_class,
            receipt: wire.receipt,
            derivation: wire.derivation,
        })
    }
}

impl EvidenceEnvelope {
    /// Create a new evidence envelope.
    pub fn new(
        claim: Claim,
        evidence_class: EvidenceClass,
        receipt: VerificationReceipt,
        derivation: Option<DerivationTree>,
    ) -> Self {
        Self {
            claim,
            evidence_class,
            receipt,
            derivation,
        }
    }

    /// Check integrity between the claim, receipt, and any attached derivation.
    ///
    /// This is a structural trust-boundary check, not a replacement for replaying the
    /// derivation under its assumptions context. Kernel-proved envelopes must carry a
    /// derivation whose root claim and digest are bound by the receipt.
    pub fn verify_integrity(&self) -> bool {
        if claim_verification_units(&self.claim).is_err() || !self.receipt.has_valid_structure() {
            return false;
        }
        if self.receipt.claim_digest != self.claim.digest()
            || self.receipt.evidence_class != self.evidence_class
        {
            return false;
        }
        // HeuristicCandidate is non-terminal by registry contract: an unverified generator
        // output must never be sealed into an EvidenceEnvelope. Refuse up front rather than
        // letting it later match the non-KernelProved no-derivation branch.
        if self.evidence_class == EvidenceClass::HeuristicCandidate {
            return false;
        }

        match &self.derivation {
            Some(derivation) => {
                if derivation_verification_units(derivation).is_err() {
                    return false;
                }
                let root_claim = derivation
                    .steps
                    .iter()
                    .find(|step| step.id == derivation.root)
                    .map(|step| &step.claim);
                root_claim == Some(&self.claim)
                    && self.receipt.derivation_digest == Some(derivation.digest())
            }
            None => {
                self.receipt.derivation_digest.is_none()
                    && self.evidence_class != EvidenceClass::KernelProved
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::Expr;
    use fsym_id::ReceiptId;
    use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};

    fn base_claim() -> Claim {
        Claim::equality(Expr::symbol("x"), Expr::symbol("x"))
    }

    fn terminal_receipt(class: EvidenceClass, verifier: &str) -> VerificationReceipt {
        VerificationReceipt::issue(
            ReceiptId::new(1).unwrap(),
            &base_claim(),
            class,
            verifier,
            1,
            None,
        )
    }

    #[test]
    fn terminal_non_kernel_proved_envelope_without_derivation_passes_integrity() {
        let envelope = EvidenceEnvelope::new(
            base_claim(),
            EvidenceClass::CertificateVerified,
            terminal_receipt(EvidenceClass::CertificateVerified, "trusted-cert-verifier"),
            None,
        );
        assert!(envelope.verify_integrity());
    }

    #[test]
    fn kernel_proved_envelope_without_derivation_fails_integrity() {
        let envelope = EvidenceEnvelope::new(
            base_claim(),
            EvidenceClass::KernelProved,
            terminal_receipt(EvidenceClass::KernelProved, "fsym-proof-kernel"),
            None,
        );
        assert!(!envelope.verify_integrity());
    }

    #[test]
    fn heuristic_candidate_envelope_never_passes_integrity() {
        // Even with a structurally valid receipt and no derivation, the
        // non-terminal HeuristicCandidate class must be refused at the
        // seal boundary: it can only be carried as a CandidateNamespace
        // entry, never as a sealed EvidenceEnvelope.
        let envelope = EvidenceEnvelope::new(
            base_claim(),
            EvidenceClass::HeuristicCandidate,
            terminal_receipt(
                EvidenceClass::HeuristicCandidate,
                "fsym-heuristic-generator",
            ),
            None,
        );
        assert!(!envelope.verify_integrity());
    }

    #[test]
    fn heuristic_candidate_with_forged_derivation_still_fails_integrity() {
        // A hostile caller attaching a fabricated derivation must not smuggle
        // a HeuristicCandidate through: the class itself is refused.
        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::Reflexivity(Expr::symbol("x")),
                claim: base_claim(),
            }],
            root: StepId(0),
        };
        let receipt = VerificationReceipt::issue(
            ReceiptId::new(2).unwrap(),
            &base_claim(),
            EvidenceClass::HeuristicCandidate,
            "fsym-heuristic-generator",
            2,
            Some(derivation.digest()),
        );
        let envelope = EvidenceEnvelope::new(
            base_claim(),
            EvidenceClass::HeuristicCandidate,
            receipt,
            Some(derivation),
        );
        assert!(!envelope.verify_integrity());
    }

    #[test]
    fn receipt_class_mismatch_fails_integrity() {
        let envelope = EvidenceEnvelope::new(
            base_claim(),
            EvidenceClass::CertificateVerified,
            terminal_receipt(EvidenceClass::OracleConformant, "fsym-profile-comparator"),
            None,
        );
        assert!(!envelope.verify_integrity());
    }
}
