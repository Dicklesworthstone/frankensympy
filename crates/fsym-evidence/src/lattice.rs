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

    #[test]
    fn user_asserted_cannot_be_promoted_to_mathematical() {
        // UserAsserted is non-mathematical by registry contract, so a
        // transition into any mathematical class (KernelProved,
        // CertificateVerified, ExactCrossChecked, CertifiedNumeric)
        // is forbidden. Pin the general gate separately from the
        // specific HeuristicCandidate / OracleConformant /
        // CertifiedNumeric cases so a regression that swaps the
        // registry field set is caught.
        for to in [
            EvidenceClass::KernelProved,
            EvidenceClass::CertificateVerified,
            EvidenceClass::ExactCrossChecked,
            EvidenceClass::CertifiedNumeric,
        ] {
            let err = validate_evidence_transition(EvidenceClass::UserAsserted, to).unwrap_err();
            assert!(matches!(err, LatticeError::IllegalPromotion { .. }));
        }
    }

    #[test]
    fn mathematical_to_mathematical_transitions_are_allowed() {
        // Among the four mathematical classes, the lattice refuses
        // two specific transitions (OracleConformant -> KernelProved
        // and CertifiedNumeric -> KernelProved). Every other
        // mathematical-to-mathematical pair (including self-loops)
        // is allowed. Pin the matrix so a future tightening is
        // intentional, not accidental.
        let mathematical = [
            EvidenceClass::KernelProved,
            EvidenceClass::CertificateVerified,
            EvidenceClass::ExactCrossChecked,
            EvidenceClass::CertifiedNumeric,
        ];
        for &from in &mathematical {
            for &to in &mathematical {
                if matches!(
                    (from, to),
                    (EvidenceClass::OracleConformant, EvidenceClass::KernelProved)
                        | (EvidenceClass::CertifiedNumeric, EvidenceClass::KernelProved)
                ) {
                    let err = validate_evidence_transition(from, to)
                        .expect_err("this transition must remain forbidden");
                    assert!(matches!(err, LatticeError::IllegalPromotion { .. }));
                } else {
                    validate_evidence_transition(from, to)
                        .expect("mathematical-to-mathematical transition must be allowed");
                }
            }
        }
    }

    #[test]
    fn heuristic_candidate_to_terminal_non_mathematical_is_also_forbidden() {
        // HeuristicCandidate is non-terminal; the lattice refuses
        // every promotion to a terminal class. My previous test
        // covered HeuristicCandidate -> KernelProved. Pin the
        // remaining terminal non-mathematical targets (OracleConformant,
        // UserAsserted) so a future relaxation cannot sneak past
        // the HeuristicCandidate gate.
        let heuristic = EvidenceClass::HeuristicCandidate;
        for to in [EvidenceClass::OracleConformant, EvidenceClass::UserAsserted] {
            let err = validate_evidence_transition(heuristic, to)
                .expect_err("HeuristicCandidate cannot become a terminal class");
            assert!(matches!(err, LatticeError::IllegalPromotion { .. }));
        }
    }
}
