//! Typed mathematical and execution outcomes for FrankenSymPy.
//!
//! Layer: L0 (outcomes). Per `docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md`
//! §4.2 this crate defines `MathOutcome<T>`, `ExecutionOutcome<T>`, evidence
//! class identifiers, and typed refusal/resource/cancellation reasons. It
//! cannot verify evidence and never maps outcomes to Python exceptions;
//! both belong to higher layers.
//!
//! The evidence-class set, its properties, and the non-evidence outcome names
//! mirror the authoritative machine registry
//! `registries/evidence_classes.toml` (schema_version 1). Unknown textual
//! class or outcome names fail closed on parsing.

#![forbid(unsafe_code)]

use std::fmt;

/// What was actually checked before an outcome was produced.
///
/// Evidence classes describe what has been verified; they are not a scalar
/// confidence ranking. Properties mirror `registries/evidence_classes.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceClass {
    /// Independently proved by the proof kernel.
    KernelProved,
    /// Verified by a claim-specific independent verifier.
    CertificateVerified,
    /// Checked by an independence-and-invariant checker.
    ExactCrossChecked,
    /// Certified by directed-rounding enclosure arithmetic.
    CertifiedNumeric,
    /// Matched a pinned upstream profile via the conformance comparator.
    OracleConformant,
    /// Asserted by the user inside a declared context.
    UserAsserted,
    /// An unverified candidate produced by a heuristic generator.
    HeuristicCandidate,
}

/// Tri-state used where the registry marks a discharge capability as
/// claim-dependent or policy-dependent rather than yes/no.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Discharge {
    Yes,
    ClaimDependent,
    PolicyDependent,
    No,
}

impl Discharge {
    /// Registry identifier string. Mirrors the `as_str` pattern on
    /// every other public enum in this crate.
    pub fn as_str(&self) -> &'static str {
        match self {
            Discharge::Yes => "yes",
            Discharge::ClaimDependent => "claim_dependent",
            Discharge::PolicyDependent => "policy_dependent",
            Discharge::No => "no",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "yes" => Self::Yes,
            "claim_dependent" => Self::ClaimDependent,
            "policy_dependent" => Self::PolicyDependent,
            "no" => Self::No,
            _ => return None,
        })
    }
}

impl EvidenceClass {
    /// Registry identifier string.
    pub fn as_str(self) -> &'static str {
        match self {
            EvidenceClass::KernelProved => "kernel_proved",
            EvidenceClass::CertificateVerified => "certificate_verified",
            EvidenceClass::ExactCrossChecked => "exact_cross_checked",
            EvidenceClass::CertifiedNumeric => "certified_numeric",
            EvidenceClass::OracleConformant => "oracle_conformant",
            EvidenceClass::UserAsserted => "user_asserted",
            EvidenceClass::HeuristicCandidate => "heuristic_candidate",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "kernel_proved" => Self::KernelProved,
            "certificate_verified" => Self::CertificateVerified,
            "exact_cross_checked" => Self::ExactCrossChecked,
            "certified_numeric" => Self::CertifiedNumeric,
            "oracle_conformant" => Self::OracleConformant,
            "user_asserted" => Self::UserAsserted,
            "heuristic_candidate" => Self::HeuristicCandidate,
            _ => return None,
        })
    }

    /// Registry field `terminal`. Only the heuristic candidate is
    /// non-terminal: it must be verified or refused before publication.
    pub fn is_terminal(self) -> bool {
        !matches!(self, EvidenceClass::HeuristicCandidate)
    }

    /// Registry field `mathematical`.
    pub fn is_mathematical(self) -> bool {
        matches!(
            self,
            EvidenceClass::KernelProved
                | EvidenceClass::CertificateVerified
                | EvidenceClass::ExactCrossChecked
                | EvidenceClass::CertifiedNumeric
        )
    }

    /// Registry field `can_discharge_exact_equality`.
    pub fn can_discharge_exact_equality(self) -> Discharge {
        match self {
            EvidenceClass::KernelProved => Discharge::Yes,
            EvidenceClass::CertificateVerified => Discharge::ClaimDependent,
            EvidenceClass::ExactCrossChecked => Discharge::PolicyDependent,
            _ => Discharge::No,
        }
    }

    /// Registry field `can_discharge_numeric_enclosure`.
    pub fn can_discharge_numeric_enclosure(self) -> Discharge {
        match self {
            EvidenceClass::CertifiedNumeric => Discharge::Yes,
            EvidenceClass::CertificateVerified => Discharge::ClaimDependent,
            _ => Discharge::No,
        }
    }

    /// Registry field `can_certify_compatibility`.
    pub fn can_certify_compatibility(self) -> bool {
        matches!(self, EvidenceClass::OracleConformant)
    }
}

impl fmt::Display for EvidenceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Non-evidence terminal states named by the registry under
/// `[non_evidence_outcomes]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NonEvidenceOutcome {
    Conditional,
    Inconclusive,
    Refused,
    Cancelled,
    TimedOut,
    ResourceExhausted,
    Unsupported,
    InternalFault,
}

impl NonEvidenceOutcome {
    /// Registry identifier string.
    pub fn as_str(self) -> &'static str {
        match self {
            NonEvidenceOutcome::Conditional => "conditional",
            NonEvidenceOutcome::Inconclusive => "inconclusive",
            NonEvidenceOutcome::Refused => "refused",
            NonEvidenceOutcome::Cancelled => "cancelled",
            NonEvidenceOutcome::TimedOut => "timed_out",
            NonEvidenceOutcome::ResourceExhausted => "resource_exhausted",
            NonEvidenceOutcome::Unsupported => "unsupported",
            NonEvidenceOutcome::InternalFault => "internal_fault",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "conditional" => Self::Conditional,
            "inconclusive" => Self::Inconclusive,
            "refused" => Self::Refused,
            "cancelled" => Self::Cancelled,
            "timed_out" => Self::TimedOut,
            "resource_exhausted" => Self::ResourceExhausted,
            "unsupported" => Self::Unsupported,
            "internal_fault" => Self::InternalFault,
            _ => return None,
        })
    }
}

impl fmt::Display for NonEvidenceOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why work was policy-refused before any mathematical evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefusalKind {
    PolicyForbidden,
    CapabilityMissing,
    DomainViolation,
    InputShapeRejected,
}

impl RefusalKind {
    /// Registry identifier string. Mirror of the `as_str` pattern on
    /// `EvidenceClass` and `NonEvidenceOutcome`.
    pub fn as_str(&self) -> &'static str {
        match self {
            RefusalKind::PolicyForbidden => "policy_forbidden",
            RefusalKind::CapabilityMissing => "capability_missing",
            RefusalKind::DomainViolation => "domain_violation",
            RefusalKind::InputShapeRejected => "input_shape_rejected",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "policy_forbidden" => Self::PolicyForbidden,
            "capability_missing" => Self::CapabilityMissing,
            "domain_violation" => Self::DomainViolation,
            "input_shape_rejected" => Self::InputShapeRejected,
            _ => return None,
        })
    }
}

/// Where in the structured-execution contract cancellation was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CancellationPoint {
    RequestCancelled,
    ParentCancelled,
    BudgetRevoked,
}

impl CancellationPoint {
    /// Registry identifier string. Mirror of the `as_str` pattern on
    /// `EvidenceClass` and `NonEvidenceOutcome`.
    pub fn as_str(&self) -> &'static str {
        match self {
            CancellationPoint::RequestCancelled => "request_cancelled",
            CancellationPoint::ParentCancelled => "parent_cancelled",
            CancellationPoint::BudgetRevoked => "budget_revoked",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "request_cancelled" => Self::RequestCancelled,
            "parent_cancelled" => Self::ParentCancelled,
            "budget_revoked" => Self::BudgetRevoked,
            _ => return None,
        })
    }
}

/// Coarse resource class reported when a budget is exhausted. The canonical
/// budget dimensions live in `fsym-budget`; this is the L0 reporting view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceClass {
    MemoryBytes,
    ComputeSteps,
    AllocationCount,
    DepthLimit,
    RandomDraws,
    TimeBudget,
}

impl ResourceClass {
    /// Registry identifier string. Mirror of the `as_str` pattern on
    /// `EvidenceClass` and `NonEvidenceOutcome`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceClass::MemoryBytes => "memory_bytes",
            ResourceClass::ComputeSteps => "compute_steps",
            ResourceClass::AllocationCount => "allocation_count",
            ResourceClass::DepthLimit => "depth_limit",
            ResourceClass::RandomDraws => "random_draws",
            ResourceClass::TimeBudget => "time_budget",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "memory_bytes" => Self::MemoryBytes,
            "compute_steps" => Self::ComputeSteps,
            "allocation_count" => Self::AllocationCount,
            "depth_limit" => Self::DepthLimit,
            "random_draws" => Self::RandomDraws,
            "time_budget" => Self::TimeBudget,
            _ => return None,
        })
    }
}

/// Internal fault kinds that quarantine affected artifacts instead of ever
/// returning a candidate as an accepted value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InternalFaultKind {
    InvariantViolation,
    SchemaViolation,
    AssertionViolation,
}

impl InternalFaultKind {
    /// Registry identifier string. Mirror of the `as_str` pattern on
    /// `EvidenceClass` and `NonEvidenceOutcome`.
    pub fn as_str(&self) -> &'static str {
        match self {
            InternalFaultKind::InvariantViolation => "invariant_violation",
            InternalFaultKind::SchemaViolation => "schema_violation",
            InternalFaultKind::AssertionViolation => "assertion_violation",
        }
    }

    /// Parses a registry identifier. Unknown names fail closed.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "invariant_violation" => Self::InvariantViolation,
            "schema_violation" => Self::SchemaViolation,
            "assertion_violation" => Self::AssertionViolation,
            _ => return None,
        })
    }
}

/// The mathematical result of requested work, including honest refusals to
/// claim one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MathOutcome<T> {
    /// A value whose attached evidence class establishes mathematical truth
    /// (class must satisfy `is_mathematical()`).
    Established {
        value: T,
        evidence: EvidenceClass,
    },
    /// A value with only non-mathematical evidence attached. It must not be
    /// published into accepted/certified paths.
    Candidate {
        value: T,
        evidence: EvidenceClass,
    },
    /// A value valid only inside its declared context; carries no evidence.
    Conditional {
        value: T,
    },
    Refused {
        kind: RefusalKind,
    },
    Cancelled {
        at: CancellationPoint,
    },
    TimedOut,
    ResourceExhausted {
        class: ResourceClass,
    },
    Unsupported {
        detail: String,
    },
    Inconclusive {
        detail: String,
    },
    InternalFault {
        kind: InternalFaultKind,
    },
}

impl<T> MathOutcome<T> {
    /// Constructs an `Established` outcome, rejecting non-mathematical
    /// evidence classes so heuristic results cannot be promoted by accident.
    pub fn established(value: T, evidence: EvidenceClass) -> Result<Self, PromotionError> {
        if evidence.is_mathematical() {
            Ok(MathOutcome::Established { value, evidence })
        } else {
            Err(PromotionError {
                attempted: evidence,
            })
        }
    }

    /// Constructs a `Candidate` outcome from non-mathematical evidence.
    pub fn candidate(value: T, evidence: EvidenceClass) -> Result<Self, PromotionError> {
        if !evidence.is_mathematical() {
            Ok(MathOutcome::Candidate { value, evidence })
        } else {
            Err(PromotionError {
                attempted: evidence,
            })
        }
    }

    /// The evidence actually attached to this outcome, if any.
    pub fn evidence(&self) -> Option<EvidenceClass> {
        match self {
            MathOutcome::Established { evidence, .. } | MathOutcome::Candidate { evidence, .. } => {
                Some(*evidence)
            }
            _ => None,
        }
    }

    /// Terminality of this outcome. A candidate carrying the
    /// `heuristic_candidate` class, a conditional value, and an inconclusive
    /// result are all non-terminal; everything else is terminal.
    pub fn is_terminal(&self) -> bool {
        match self {
            MathOutcome::Candidate { evidence, .. } => evidence.is_terminal(),
            MathOutcome::Conditional { .. } | MathOutcome::Inconclusive { .. } => false,
            _ => true,
        }
    }

    /// Maps the value type, preserving outcome structure.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> MathOutcome<U> {
        match self {
            MathOutcome::Established { value, evidence } => MathOutcome::Established {
                value: f(value),
                evidence,
            },
            MathOutcome::Candidate { value, evidence } => MathOutcome::Candidate {
                value: f(value),
                evidence,
            },
            MathOutcome::Conditional { value } => MathOutcome::Conditional { value: f(value) },
            MathOutcome::Refused { kind } => MathOutcome::Refused { kind },
            MathOutcome::Cancelled { at } => MathOutcome::Cancelled { at },
            MathOutcome::TimedOut => MathOutcome::TimedOut,
            MathOutcome::ResourceExhausted { class } => MathOutcome::ResourceExhausted { class },
            MathOutcome::Unsupported { detail } => MathOutcome::Unsupported { detail },
            MathOutcome::Inconclusive { detail } => MathOutcome::Inconclusive { detail },
            MathOutcome::InternalFault { kind } => MathOutcome::InternalFault { kind },
        }
    }
}

/// Attempted construction of an outcome with an evidence class whose strength
/// does not match the outcome variant (e.g. promoting a heuristic candidate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotionError {
    /// The offending class.
    pub attempted: EvidenceClass,
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot attach evidence class {} to this outcome",
            self.attempted
        )
    }
}

impl std::error::Error for PromotionError {}

/// Execution-level envelope returned to callers once all controlled children
/// have been drained.
///
/// The accounting fields exist so that the structured-concurrency contract
/// ("after return: controlled orphan count zero, no unverified publication")
/// is observable on the value itself rather than in prose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionOutcome<T> {
    /// The mathematical outcome of the request.
    pub math: MathOutcome<T>,
    /// Controlled children that outlived the owning region after drain. This
    /// MUST be zero for a clean return; nonzero values quarantine the run.
    pub controlled_orphans: u64,
    /// Whether any candidate reached a publication surface without
    /// verification. Must be false for a clean return.
    pub published_unverified: bool,
}

impl<T> ExecutionOutcome<T> {
    /// Wraps a math outcome with clean accounting (zero orphans, nothing
    /// published unverified).
    pub fn finish(math: MathOutcome<T>) -> Self {
        ExecutionOutcome {
            math,
            controlled_orphans: 0,
            published_unverified: false,
        }
    }

    /// True when the execution contract holds: no orphaned children and no
    /// unverified publication.
    pub fn is_clean_return(&self) -> bool {
        self.controlled_orphans == 0 && !self.published_unverified
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registry_class_roundtrips() {
        let all = [
            EvidenceClass::KernelProved,
            EvidenceClass::CertificateVerified,
            EvidenceClass::ExactCrossChecked,
            EvidenceClass::CertifiedNumeric,
            EvidenceClass::OracleConformant,
            EvidenceClass::UserAsserted,
            EvidenceClass::HeuristicCandidate,
        ];
        for class in all {
            assert_eq!(EvidenceClass::parse(class.as_str()), Some(class));
        }
        // Fail closed on unknown names.
        assert!(EvidenceClass::parse("definitely_proved").is_none());
        assert!(EvidenceClass::parse("").is_none());
    }

    #[test]
    fn registry_mathematical_flags_match() {
        use EvidenceClass::*;
        let mathematical = [
            KernelProved,
            CertificateVerified,
            ExactCrossChecked,
            CertifiedNumeric,
        ];
        let non_mathematical = [OracleConformant, UserAsserted, HeuristicCandidate];
        for c in mathematical {
            assert!(c.is_mathematical(), "{c} should be mathematical");
        }
        for c in non_mathematical {
            assert!(!c.is_mathematical(), "{c} must not be mathematical");
        }
    }

    #[test]
    fn only_heuristic_candidate_is_nonterminal() {
        for class in [
            EvidenceClass::KernelProved,
            EvidenceClass::CertificateVerified,
            EvidenceClass::ExactCrossChecked,
            EvidenceClass::CertifiedNumeric,
            EvidenceClass::OracleConformant,
            EvidenceClass::UserAsserted,
        ] {
            assert!(class.is_terminal());
        }
        assert!(!EvidenceClass::HeuristicCandidate.is_terminal());
    }

    #[test]
    fn discharge_predicates_mirror_registry() {
        // can_discharge_exact_equality rows.
        assert_eq!(
            EvidenceClass::KernelProved.can_discharge_exact_equality(),
            Discharge::Yes
        );
        assert_eq!(
            EvidenceClass::CertificateVerified.can_discharge_exact_equality(),
            Discharge::ClaimDependent
        );
        assert_eq!(
            EvidenceClass::ExactCrossChecked.can_discharge_exact_equality(),
            Discharge::PolicyDependent
        );
        for c in [
            EvidenceClass::CertifiedNumeric,
            EvidenceClass::OracleConformant,
            EvidenceClass::UserAsserted,
            EvidenceClass::HeuristicCandidate,
        ] {
            assert_eq!(c.can_discharge_exact_equality(), Discharge::No);
        }

        // can_discharge_numeric_enclosure rows.
        assert_eq!(
            EvidenceClass::CertifiedNumeric.can_discharge_numeric_enclosure(),
            Discharge::Yes
        );
        assert_eq!(
            EvidenceClass::CertificateVerified.can_discharge_numeric_enclosure(),
            Discharge::ClaimDependent
        );
        for c in [
            EvidenceClass::KernelProved,
            EvidenceClass::ExactCrossChecked,
            EvidenceClass::OracleConformant,
            EvidenceClass::UserAsserted,
            EvidenceClass::HeuristicCandidate,
        ] {
            assert_eq!(c.can_discharge_numeric_enclosure(), Discharge::No);
        }

        // can_certify_compatibility rows.
        assert!(EvidenceClass::OracleConformant.can_certify_compatibility());
        for c in [
            EvidenceClass::KernelProved,
            EvidenceClass::CertificateVerified,
            EvidenceClass::ExactCrossChecked,
            EvidenceClass::CertifiedNumeric,
            EvidenceClass::UserAsserted,
            EvidenceClass::HeuristicCandidate,
        ] {
            assert!(!c.can_certify_compatibility());
        }
    }

    #[test]
    fn every_non_evidence_outcome_roundtrips() {
        let all = [
            NonEvidenceOutcome::Conditional,
            NonEvidenceOutcome::Inconclusive,
            NonEvidenceOutcome::Refused,
            NonEvidenceOutcome::Cancelled,
            NonEvidenceOutcome::TimedOut,
            NonEvidenceOutcome::ResourceExhausted,
            NonEvidenceOutcome::Unsupported,
            NonEvidenceOutcome::InternalFault,
        ];
        for o in all {
            assert_eq!(NonEvidenceOutcome::parse(o.as_str()), Some(o));
        }
        assert!(NonEvidenceOutcome::parse("proved").is_none());
    }

    #[test]
    fn constructor_guards_block_promotion() {
        // A heuristic result must not be constructed as Established.
        let err = MathOutcome::<u8>::established(1, EvidenceClass::HeuristicCandidate).unwrap_err();
        assert_eq!(err.attempted, EvidenceClass::HeuristicCandidate);

        // Mathematical evidence must not be downgraded into a Candidate slot.
        let err = MathOutcome::<u8>::candidate(1, EvidenceClass::KernelProved).unwrap_err();
        assert_eq!(err.attempted, EvidenceClass::KernelProved);

        // Oracle parity must not become mathematical establishment.
        assert!(MathOutcome::<u8>::established(1, EvidenceClass::OracleConformant).is_err());
    }

    #[test]
    fn terminality_rules() {
        let heuristic = MathOutcome::<u8>::candidate(1, EvidenceClass::HeuristicCandidate).unwrap();
        assert!(!heuristic.is_terminal());
        assert!(!MathOutcome::<u8>::Conditional { value: 1 }.is_terminal());
        assert!(
            !MathOutcome::<u8>::Inconclusive {
                detail: "split".into()
            }
            .is_terminal()
        );
        assert!(MathOutcome::<u8>::TimedOut.is_terminal());
        let proved = MathOutcome::<u8>::established(1, EvidenceClass::KernelProved).unwrap();
        assert!(proved.is_terminal());
    }

    #[test]
    fn map_preserves_structure_and_evidence() {
        let proved = MathOutcome::<u32>::established(7, EvidenceClass::KernelProved).unwrap();
        let mapped = proved.map(|v| v * 2);
        assert_eq!(
            mapped.evidence(),
            Some(EvidenceClass::KernelProved),
            "mapping a value must never change its evidence class"
        );

        let refused = MathOutcome::<u32>::Refused {
            kind: RefusalKind::PolicyForbidden,
        };
        assert!(matches!(refused.map(|v| v), MathOutcome::Refused { .. }));
    }

    #[test]
    fn execution_contract_is_observable() {
        let clean = ExecutionOutcome::finish(MathOutcome::<u8>::TimedOut);
        assert!(clean.is_clean_return());

        let dirty = ExecutionOutcome::<u8> {
            math: MathOutcome::TimedOut,
            controlled_orphans: 2,
            published_unverified: false,
        };
        assert!(!dirty.is_clean_return());

        let leaky = ExecutionOutcome::<u8> {
            math: MathOutcome::TimedOut,
            controlled_orphans: 0,
            published_unverified: true,
        };
        assert!(!leaky.is_clean_return());
    }

    #[test]
    fn sampled_agreement_cannot_become_exact_identity() {
        // Prohibited promotion from registries/evidence_classes.toml:
        // sampled numeric agreement must stay a candidate at best.
        let sampled = MathOutcome::<u8>::candidate(3, EvidenceClass::UserAsserted).unwrap();
        assert_ne!(sampled.evidence(), Some(EvidenceClass::KernelProved));
        assert!(!sampled.evidence().unwrap().is_mathematical());
    }

    #[test]
    fn refusal_kind_roundtrips_through_registry_identifier() {
        let all = [
            RefusalKind::PolicyForbidden,
            RefusalKind::CapabilityMissing,
            RefusalKind::DomainViolation,
            RefusalKind::InputShapeRejected,
        ];
        for kind in all.iter() {
            assert_eq!(RefusalKind::parse(kind.as_str()), Some(kind.clone()));
        }
        // Fail closed on unknown names and the empty string.
        assert!(RefusalKind::parse("policy_breach").is_none());
        assert!(RefusalKind::parse("").is_none());
    }

    #[test]
    fn cancellation_point_roundtrips_through_registry_identifier() {
        let all = [
            CancellationPoint::RequestCancelled,
            CancellationPoint::ParentCancelled,
            CancellationPoint::BudgetRevoked,
        ];
        for point in all.iter() {
            assert_eq!(CancellationPoint::parse(point.as_str()), Some(*point));
        }
        assert!(CancellationPoint::parse("timeout").is_none());
        assert!(CancellationPoint::parse("").is_none());
    }

    #[test]
    fn resource_class_roundtrips_through_registry_identifier() {
        let all = [
            ResourceClass::MemoryBytes,
            ResourceClass::ComputeSteps,
            ResourceClass::AllocationCount,
            ResourceClass::DepthLimit,
            ResourceClass::RandomDraws,
            ResourceClass::TimeBudget,
        ];
        for class in all.iter() {
            assert_eq!(ResourceClass::parse(class.as_str()), Some(*class));
        }
        assert!(ResourceClass::parse("bytes").is_none());
        assert!(ResourceClass::parse("").is_none());
    }

    #[test]
    fn internal_fault_kind_roundtrips_through_registry_identifier() {
        let all = [
            InternalFaultKind::InvariantViolation,
            InternalFaultKind::SchemaViolation,
            InternalFaultKind::AssertionViolation,
        ];
        for kind in all.iter() {
            assert_eq!(InternalFaultKind::parse(kind.as_str()), Some(*kind));
        }
        assert!(InternalFaultKind::parse("panic").is_none());
        assert!(InternalFaultKind::parse("").is_none());
    }

    #[test]
    fn discharge_roundtrips_through_registry_identifier() {
        let all = [
            Discharge::Yes,
            Discharge::ClaimDependent,
            Discharge::PolicyDependent,
            Discharge::No,
        ];
        for d in all.iter() {
            assert_eq!(Discharge::parse(d.as_str()), Some(*d));
        }
        assert!(Discharge::parse("maybe").is_none());
        assert!(Discharge::parse("").is_none());
    }
}
