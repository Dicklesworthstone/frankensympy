//! Cryptographic verification receipts for FrankenSymPy (WS06).
//!
//! Layer: L2 (evidence and receipts).
//! A receipt is an unforgeable witness that an independent verifier checked a specific
//! mathematical claim under a specific context.

#![forbid(unsafe_code)]

use fsym_id::ReceiptId;
use fsym_outcome::EvidenceClass;
use fsym_proof_kernel::Claim;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Unforgeable witness emitted when an independent verifier accepts a claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReceipt {
    pub receipt_id: ReceiptId,
    pub claim_digest: [u8; 32],
    pub evidence_class: EvidenceClass,
    pub verifier_name: String,
    pub timestamp_seq: u64,
    pub derivation_digest: Option<[u8; 32]>,
}

#[derive(Serialize, Deserialize)]
struct ReceiptWire {
    receipt_id: u64,
    claim_digest: [u8; 32],
    evidence_class: String,
    verifier_name: String,
    timestamp_seq: u64,
    derivation_digest: Option<[u8; 32]>,
}

impl Serialize for VerificationReceipt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = ReceiptWire {
            receipt_id: self.receipt_id.raw(),
            claim_digest: self.claim_digest,
            evidence_class: self.evidence_class.as_str().to_string(),
            verifier_name: self.verifier_name.clone(),
            timestamp_seq: self.timestamp_seq,
            derivation_digest: self.derivation_digest,
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for VerificationReceipt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ReceiptWire::deserialize(deserializer)?;
        let receipt_id =
            ReceiptId::new(wire.receipt_id).map_err(|e| serde::de::Error::custom(e.to_string()))?;
        let evidence_class = EvidenceClass::parse(&wire.evidence_class).ok_or_else(|| {
            serde::de::Error::custom(format!("unknown evidence class `{}`", wire.evidence_class))
        })?;
        Ok(Self {
            receipt_id,
            claim_digest: wire.claim_digest,
            evidence_class,
            verifier_name: wire.verifier_name,
            timestamp_seq: wire.timestamp_seq,
            derivation_digest: wire.derivation_digest,
        })
    }
}

impl VerificationReceipt {
    /// Issue a new receipt for a verified claim.
    pub fn issue(
        receipt_id: ReceiptId,
        claim: &Claim,
        evidence_class: EvidenceClass,
        verifier_name: impl Into<String>,
        timestamp_seq: u64,
        derivation_digest: Option<[u8; 32]>,
    ) -> Self {
        Self {
            receipt_id,
            claim_digest: claim.digest(),
            evidence_class,
            verifier_name: verifier_name.into(),
            timestamp_seq,
            derivation_digest,
        }
    }

    /// Canonical BLAKE3 digest of this receipt.
    pub fn digest(&self) -> [u8; 32] {
        let serialized = serde_json::to_vec(self).expect("receipt is serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.receipt.v1:");
        hasher.update(&serialized);
        *hasher.finalize().as_bytes()
    }
}
