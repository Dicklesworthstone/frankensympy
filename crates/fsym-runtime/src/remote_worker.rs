//! Untrusted remote worker execution lane and coordinator verification (WS16 / architecture §7.8).
//!
//! Rule (§7.5, §7.8): Remote workers only generate candidates. The coordinator MUST
//! independently verify any candidate derivation before accepting it. Worker signatures
//! or votes NEVER substitute for mathematical proof.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::Expr;
use fsym_proof_kernel::{
    Claim, DerivationTree, KernelError, claim_verification_units, derivation_verification_units,
    expression_verification_units, verify_derivation_independent,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Maximum JSON wire size accepted for one untrusted remote candidate.
///
/// This transport cap is deliberately independent of the proof kernel's in-memory
/// traversal limits. It bounds allocation during decoding before typed preflights run.
pub const MAX_REMOTE_CANDIDATE_BYTES: usize = 1024 * 1024;

/// Privacy-safe category for a rejected independent verification.
///
/// Kernel errors can contain complete expressions and symbol names. Remote-boundary
/// diagnostics expose only a stable category so an untrusted payload cannot amplify logs
/// or disclose private formulas through an error string.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RemoteVerificationFailure {
    #[error("unknown proof step")]
    UnknownStep,
    #[error("invalid proof step reference")]
    InvalidStepReference,
    #[error("proof rule mismatch")]
    RuleMismatch,
    #[error("transitivity mismatch")]
    TransitivityMismatch,
    #[error("symmetry requires equality")]
    SymmetryRequiresEquality,
    #[error("invalid congruence")]
    InvalidCongruence,
    #[error("invalid substitution")]
    InvalidSubstitution,
    #[error("required predicate is not entailed")]
    PredicateNotEntailed,
    #[error("required domain is not entailed")]
    DomainNotEntailed,
    #[error("invalid definitional reduction")]
    InvalidDefinitionalReduction,
    #[error("derived claim does not match the declared claim")]
    ClaimDiscrepancy,
    #[error("certificate lemma has no trusted verifier")]
    UnverifiedCertificateLemma,
    #[error("invalid certificate lemma")]
    InvalidCertificateLemma,
    #[error("derivation exceeds a trusted verifier limit")]
    DerivationLimitExceeded,
    #[error("proof step identifier space is exhausted")]
    StepIdExhausted,
    #[error("verifier budget failure")]
    Budget,
}

impl From<KernelError> for RemoteVerificationFailure {
    fn from(error: KernelError) -> Self {
        match error {
            KernelError::UnknownStep(_) => Self::UnknownStep,
            KernelError::InvalidStepReference(_) => Self::InvalidStepReference,
            KernelError::RuleMismatch(_) => Self::RuleMismatch,
            KernelError::TransitivityMismatch { .. } => Self::TransitivityMismatch,
            KernelError::SymmetryRequiresEquality(_) => Self::SymmetryRequiresEquality,
            KernelError::InvalidCongruence(_) => Self::InvalidCongruence,
            KernelError::InvalidSubstitution(_) => Self::InvalidSubstitution,
            KernelError::PredicateNotEntailed { .. } => Self::PredicateNotEntailed,
            KernelError::DomainNotEntailed { .. } => Self::DomainNotEntailed,
            KernelError::InvalidDefinitionalReduction { .. } => Self::InvalidDefinitionalReduction,
            KernelError::ClaimDiscrepancy { .. } => Self::ClaimDiscrepancy,
            KernelError::UnverifiedCertificateLemma { .. } => Self::UnverifiedCertificateLemma,
            KernelError::InvalidCertificateLemma { .. } => Self::InvalidCertificateLemma,
            KernelError::DerivationLimitExceeded { .. } => Self::DerivationLimitExceeded,
            KernelError::StepIdExhausted => Self::StepIdExhausted,
            KernelError::Budget(_) => Self::Budget,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RemoteWorkerError {
    #[error("Candidate derivation verification failed: {0}")]
    VerificationFailed(RemoteVerificationFailure),
    #[error("Untrusted worker claim forgery: claimed result does not match verified claim")]
    ClaimForgery,
    #[error("Untrusted worker response does not answer the assigned task")]
    TaskMismatch,
    #[error("Worker timeout or communication fault")]
    WorkerFault,
    #[error("Payload schema or integrity corruption")]
    CorruptedPayload,
    #[error("Remote candidate payload exceeds the {limit}-byte wire limit")]
    PayloadTooLarge { limit: usize },
    #[error("Coordinator assignment exceeds trusted verification bounds")]
    InvalidAssignment,
}

/// A candidate produced by an untrusted remote worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RemoteCandidate {
    pub worker_id: String,
    pub task_id: u64,
    pub result: Expr,
    pub claim: Claim,
    pub derivation: DerivationTree,
    pub worker_signature: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RemoteCandidateWire {
    worker_id: String,
    task_id: u64,
    result: Expr,
    claim: Claim,
    derivation: DerivationTree,
    worker_signature: Vec<u8>,
}

impl RemoteCandidate {
    /// Decode one bounded JSON response from an untrusted worker.
    ///
    /// `RemoteCandidate` intentionally does not implement [`Deserialize`]: callers must
    /// enter through this size-bounded decoder rather than materializing an arbitrarily
    /// large worker-controlled graph before validation.
    pub fn decode_json(bytes: &[u8]) -> Result<Self, RemoteWorkerError> {
        if bytes.len() > MAX_REMOTE_CANDIDATE_BYTES {
            return Err(RemoteWorkerError::PayloadTooLarge {
                limit: MAX_REMOTE_CANDIDATE_BYTES,
            });
        }
        let wire: RemoteCandidateWire =
            serde_json::from_slice(bytes).map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        let candidate = Self {
            worker_id: wire.worker_id,
            task_id: wire.task_id,
            result: wire.result,
            claim: wire.claim,
            derivation: wire.derivation,
            worker_signature: wire.worker_signature,
        };
        candidate.preflight_fields()?;
        derivation_verification_units(&candidate.derivation)
            .map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        Ok(candidate)
    }

    fn preflight_fields(&self) -> Result<(), RemoteWorkerError> {
        if self.worker_id.is_empty()
            || self.worker_id.len() > 256
            || self.worker_signature.len() > 4_096
        {
            return Err(RemoteWorkerError::CorruptedPayload);
        }
        claim_verification_units(&self.claim).map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        expression_verification_units(&self.result)
            .map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        Ok(())
    }
}

/// An accepted result certified by the local coordinator's independent verifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedAcceptedResult {
    task_id: u64,
    result: Expr,
    claim: Claim,
    context_digest: [u8; 32],
    verifier_receipt_digest: [u8; 32],
}

impl VerifiedAcceptedResult {
    pub fn task_id(&self) -> u64 {
        self.task_id
    }

    pub fn result(&self) -> &Expr {
        &self.result
    }

    pub fn claim(&self) -> &Claim {
        &self.claim
    }

    pub fn context_digest(&self) -> [u8; 32] {
        self.context_digest
    }

    pub fn verifier_receipt_digest(&self) -> [u8; 32] {
        self.verifier_receipt_digest
    }
}

/// Coordinator supervising untrusted remote workers.
pub struct CoordinatorVerifier {
    task_id: u64,
    expected_claim: Claim,
    context: Arc<ImmutableAssumptionsSnapshot>,
}

impl CoordinatorVerifier {
    pub fn new(
        task_id: u64,
        expected_claim: Claim,
        context: Arc<ImmutableAssumptionsSnapshot>,
    ) -> Self {
        Self {
            task_id,
            expected_claim,
            context,
        }
    }

    /// Immutable assumptions snapshot bound to this coordinator assignment.
    pub fn context(&self) -> &ImmutableAssumptionsSnapshot {
        &self.context
    }

    /// Evaluates and verifies a candidate from an untrusted remote worker.
    ///
    /// Fails closed if the derivation is invalid, context predicates are unproven,
    /// or the claim is forged.
    pub fn verify_remote_candidate(
        &self,
        candidate: &RemoteCandidate,
    ) -> Result<VerifiedAcceptedResult, RemoteWorkerError> {
        // 1. Reject responses that cannot possibly answer this assignment before spending
        // verifier work on an untrusted derivation.
        candidate.preflight_fields()?;
        claim_verification_units(&self.expected_claim)
            .map_err(|_| RemoteWorkerError::InvalidAssignment)?;
        if candidate.task_id != self.task_id || candidate.claim != self.expected_claim {
            return Err(RemoteWorkerError::TaskMismatch);
        }
        if claimed_result(&candidate.claim) != &candidate.result {
            return Err(RemoteWorkerError::ClaimForgery);
        }

        // 2. Independent stateless verification of the derivation tree. The proof kernel's
        // trust-boundary preflight caps steps, expression depth, nodes, text, and numeric limbs.
        let verified_claim = verify_derivation_independent(&candidate.derivation, &self.context)
            .map_err(|error| RemoteWorkerError::VerificationFailed(error.into()))?;

        // 3. Bind both the worker's claim and returned value to the exact verified root.
        // Derivation export order is not itself a statement about which step is the root.
        if verified_claim != candidate.claim || claimed_result(&verified_claim) != &candidate.result
        {
            return Err(RemoteWorkerError::ClaimForgery);
        }
        if verified_claim != self.expected_claim {
            return Err(RemoteWorkerError::TaskMismatch);
        }

        // 4. Compute a structural BLAKE3 verifier receipt. This digest binds the accepted fields;
        // it is not a worker signature or authentication token.
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.coordinator.verified.v1:");
        hasher.update(&candidate.task_id.to_le_bytes());
        let claim_bytes = serde_json::to_vec(&candidate.claim)
            .map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        let result_bytes = serde_json::to_vec(&candidate.result)
            .map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        hasher.update(&claim_bytes);
        hasher.update(&result_bytes);
        hasher.update(&candidate.derivation.digest());
        hasher.update(&self.context.digest());
        let verifier_receipt_digest = *hasher.finalize().as_bytes();

        Ok(VerifiedAcceptedResult {
            task_id: candidate.task_id,
            result: candidate.result.clone(),
            claim: candidate.claim.clone(),
            context_digest: self.context.digest(),
            verifier_receipt_digest,
        })
    }
}

fn claimed_result(claim: &Claim) -> &Expr {
    match claim {
        Claim::Equality { rhs, .. } | Claim::AlgebraicIdentity { rhs, .. } => rhs,
        Claim::PredicateHold { expr, .. }
        | Claim::DomainMembership { expr, .. }
        | Claim::NonZero(expr) => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_proof_kernel::ProofKernel;

    #[test]
    fn rejects_result_not_bound_to_verified_claim() {
        let context = ImmutableAssumptionsSnapshot::empty();
        let x = Expr::symbol("x");
        let expected_claim = Claim::equality(x.clone(), x.clone());
        let coordinator = CoordinatorVerifier::new(999, expected_claim, context);
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let root = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
        let derivation = kernel.export_derivation(root).unwrap();

        let candidate = RemoteCandidate {
            worker_id: "untrusted-worker".to_string(),
            task_id: 999,
            result: Expr::from_i64(999),
            claim: Claim::equality(x.clone(), x),
            derivation,
            worker_signature: vec![0x13, 0x37],
        };

        assert_eq!(
            coordinator.verify_remote_candidate(&candidate),
            Err(RemoteWorkerError::ClaimForgery)
        );
    }

    #[test]
    fn rejects_valid_but_irrelevant_remote_result() {
        let context = ImmutableAssumptionsSnapshot::empty();
        let requested = Expr::symbol("requested");
        let coordinator =
            CoordinatorVerifier::new(7, Claim::equality(requested.clone(), requested), context);
        let irrelevant = Expr::symbol("irrelevant");
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let root = kernel
            .prove_reflexivity(irrelevant.clone(), &mut Unbounded)
            .unwrap();
        let candidate = RemoteCandidate {
            worker_id: "untrusted-worker".to_string(),
            task_id: 7,
            result: irrelevant.clone(),
            claim: Claim::equality(irrelevant.clone(), irrelevant),
            derivation: kernel.export_derivation(root).unwrap(),
            worker_signature: vec![],
        };

        assert_eq!(
            coordinator.verify_remote_candidate(&candidate),
            Err(RemoteWorkerError::TaskMismatch)
        );
    }

    #[test]
    fn rejects_wrong_task_before_inspecting_derivation() {
        let context = ImmutableAssumptionsSnapshot::empty();
        let x = Expr::symbol("x");
        let expected = Claim::equality(x.clone(), x.clone());
        let coordinator = CoordinatorVerifier::new(7, expected.clone(), context);
        let candidate = RemoteCandidate {
            worker_id: "untrusted-worker".to_string(),
            task_id: 8,
            result: x,
            claim: expected,
            derivation: DerivationTree {
                steps: Vec::new(),
                root: fsym_proof_kernel::StepId(u32::MAX),
            },
            worker_signature: Vec::new(),
        };

        assert_eq!(
            coordinator.verify_remote_candidate(&candidate),
            Err(RemoteWorkerError::TaskMismatch),
            "assignment mismatch must win over malformed-proof diagnostics"
        );
    }

    #[test]
    fn refuses_oversized_coordinator_assignment_before_comparison() {
        let context = ImmutableAssumptionsSnapshot::empty();
        let mut deep = Expr::symbol("x");
        for _ in 0..300 {
            deep = Expr::Add(vec![deep]);
        }
        let coordinator = CoordinatorVerifier::new(7, Claim::equality(deep.clone(), deep), context);
        let x = Expr::symbol("x");
        let candidate = RemoteCandidate {
            worker_id: "untrusted-worker".to_string(),
            task_id: 7,
            result: x.clone(),
            claim: Claim::equality(x.clone(), x),
            derivation: DerivationTree {
                steps: Vec::new(),
                root: fsym_proof_kernel::StepId(u32::MAX),
            },
            worker_signature: Vec::new(),
        };

        assert_eq!(
            coordinator.verify_remote_candidate(&candidate),
            Err(RemoteWorkerError::InvalidAssignment)
        );
    }

    #[test]
    fn bounded_wire_decoder_roundtrips_a_valid_candidate() {
        let x = Expr::symbol("x");
        let claim = Claim::equality(x.clone(), x.clone());
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let root = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
        let candidate = RemoteCandidate {
            worker_id: "worker-1".to_string(),
            task_id: 42,
            result: x,
            claim,
            derivation: kernel.export_derivation(root).unwrap(),
            worker_signature: vec![1, 2, 3],
        };
        let encoded = serde_json::to_vec(&candidate).unwrap();

        assert_eq!(RemoteCandidate::decode_json(&encoded), Ok(candidate));
    }

    #[test]
    fn bounded_wire_decoder_rejects_oversized_and_unknown_fields() {
        let oversized = vec![b' '; MAX_REMOTE_CANDIDATE_BYTES + 1];
        assert_eq!(
            RemoteCandidate::decode_json(&oversized),
            Err(RemoteWorkerError::PayloadTooLarge {
                limit: MAX_REMOTE_CANDIDATE_BYTES
            })
        );

        let x = Expr::symbol("x");
        let claim = Claim::equality(x.clone(), x.clone());
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let root = kernel.prove_reflexivity(x.clone(), &mut Unbounded).unwrap();
        let candidate = RemoteCandidate {
            worker_id: "worker-1".to_string(),
            task_id: 1,
            result: x,
            claim,
            derivation: kernel.export_derivation(root).unwrap(),
            worker_signature: Vec::new(),
        };
        let mut unknown_field = serde_json::to_value(candidate).unwrap();
        unknown_field
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_string(), serde_json::Value::Bool(true));
        let unknown_field = serde_json::to_vec(&unknown_field).unwrap();
        assert_eq!(
            RemoteCandidate::decode_json(&unknown_field),
            Err(RemoteWorkerError::CorruptedPayload)
        );
    }

    #[test]
    fn verifier_errors_do_not_echo_untrusted_formulas() {
        let secret = Expr::symbol("private_formula_name_that_must_not_reach_diagnostics");
        let forged_result = Expr::from_i64(7);
        let forged_claim = Claim::equality(secret.clone(), forged_result.clone());
        let coordinator = CoordinatorVerifier::new(
            77,
            forged_claim.clone(),
            ImmutableAssumptionsSnapshot::empty(),
        );
        let mut kernel = ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone());
        let root = kernel
            .prove_reflexivity(secret.clone(), &mut Unbounded)
            .unwrap();
        let mut derivation = kernel.export_derivation(root).unwrap();
        derivation.steps[0].claim = forged_claim.clone();
        let candidate = RemoteCandidate {
            worker_id: "worker-1".to_string(),
            task_id: 77,
            result: forged_result,
            claim: forged_claim,
            derivation,
            worker_signature: Vec::new(),
        };

        let error = coordinator.verify_remote_candidate(&candidate).unwrap_err();
        assert_eq!(
            error,
            RemoteWorkerError::VerificationFailed(RemoteVerificationFailure::ClaimDiscrepancy)
        );
        assert!(!error.to_string().contains("private_formula_name"));
    }
}
