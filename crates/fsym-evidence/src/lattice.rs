//! Non-conversion lattice rules for mathematical and operational evidence (WS06).
//!
//! Layer: L2 (evidence lattice).
//! Prevents illegal promotions:
//! - HeuristicCandidate -> KernelProved (forbidden)
//! - OracleConformant -> Mathematical truth (forbidden)
//! - Sampled numerical agreement -> Exact identity (forbidden)

#![forbid(unsafe_code)]

use fsym_outcome::EvidenceClass;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LatticeError {
    #[error("Illegal evidence promotion: `{from:?}` cannot be promoted to `{to:?}`")]
    IllegalPromotion {
        from: EvidenceClass,
        to: EvidenceClass,
    },
}

/// Verifies whether an evidence class transition is permissible under the evidence lattice rules.
pub fn validate_evidence_transition(
    from: EvidenceClass,
    to: EvidenceClass,
) -> Result<(), LatticeError> {
    if from == to {
        return Ok(());
    }

    // Mathematical proofs cannot be fabricated from non-mathematical or heuristic classes
    if !from.is_mathematical() && to.is_mathematical() {
        return Err(LatticeError::IllegalPromotion { from, to });
    }

    // Heuristic candidates cannot be directly converted to any terminal class without verification
    if from == EvidenceClass::HeuristicCandidate && to.is_terminal() {
        return Err(LatticeError::IllegalPromotion { from, to });
    }

    // Oracle conformity is not mathematical proof
    if from == EvidenceClass::OracleConformant && to == EvidenceClass::KernelProved {
        return Err(LatticeError::IllegalPromotion { from, to });
    }

    // Certified numeric cannot discharge exact algebraic equality
    if from == EvidenceClass::CertifiedNumeric && to == EvidenceClass::KernelProved {
        return Err(LatticeError::IllegalPromotion { from, to });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_cannot_promote_to_kernel_proved() {
        let err = validate_evidence_transition(
            EvidenceClass::HeuristicCandidate,
            EvidenceClass::KernelProved,
        )
        .unwrap_err();
        assert!(matches!(err, LatticeError::IllegalPromotion { .. }));
    }

    #[test]
    fn oracle_conformant_cannot_promote_to_kernel_proved() {
        let err = validate_evidence_transition(
            EvidenceClass::OracleConformant,
            EvidenceClass::KernelProved,
        )
        .unwrap_err();
        assert!(matches!(err, LatticeError::IllegalPromotion { .. }));
    }

    #[test]
    fn certified_numeric_cannot_promote_to_kernel_proved() {
        let err = validate_evidence_transition(
            EvidenceClass::CertifiedNumeric,
            EvidenceClass::KernelProved,
        )
        .unwrap_err();
        assert!(matches!(err, LatticeError::IllegalPromotion { .. }));
    }

    #[test]
    fn identical_classes_are_valid() {
        assert!(
            validate_evidence_transition(EvidenceClass::KernelProved, EvidenceClass::KernelProved)
                .is_ok()
        );
    }
}
