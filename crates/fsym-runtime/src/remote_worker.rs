//! Untrusted remote worker execution lane and coordinator verification (WS16 / architecture §7.8).
//!
//! Rule (§7.5, §7.8): Remote workers only generate candidates. The coordinator MUST
//! independently verify any candidate derivation before accepting it. Worker signatures
//! or votes NEVER substitute for mathematical proof.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::Expr;
use fsym_proof_kernel::{
    Claim, DerivationTree, claim_verification_units, expression_verification_units,
    verify_derivation_independent,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RemoteWorkerError {
    #[error("Candidate derivation verification failed: {0}")]
    VerificationFailed(String),
    #[error("Untrusted worker claim forgery: claimed result does not match verified claim")]
    ClaimForgery,
    #[error("Untrusted worker response does not answer the assigned task")]
    TaskMismatch,
    #[error("Worker timeout or communication fault")]
    WorkerFault,
    #[error("Payload schema or integrity corruption")]
    CorruptedPayload,
    #[error("Coordinator assignment exceeds trusted verification bounds")]
    InvalidAssignment,
}

/// A candidate produced by an untrusted remote worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteCandidate {
    pub worker_id: String,
    pub task_id: u64,
    pub result: Expr,
    pub claim: Claim,
    pub derivation: DerivationTree,
    pub worker_signature: Vec<u8>,
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
    pub context: Arc<ImmutableAssumptionsSnapshot>,
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
        if candidate.worker_id.is_empty()
            || candidate.worker_id.len() > 256
            || candidate.worker_signature.len() > 4_096
        {
            return Err(RemoteWorkerError::CorruptedPayload);
        }
        claim_verification_units(&self.expected_claim)
            .map_err(|_| RemoteWorkerError::InvalidAssignment)?;
        claim_verification_units(&candidate.claim)
            .map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        expression_verification_units(&candidate.result)
            .map_err(|_| RemoteWorkerError::CorruptedPayload)?;
        if candidate.task_id != self.task_id || candidate.claim != self.expected_claim {
            return Err(RemoteWorkerError::TaskMismatch);
        }
        if claimed_result(&candidate.claim) != &candidate.result {
            return Err(RemoteWorkerError::ClaimForgery);
        }

        // 2. Independent stateless verification of the derivation tree. The proof kernel's
        // trust-boundary preflight caps steps, expression depth, nodes, text, and numeric limbs.
        let verified_claim = verify_derivation_independent(&candidate.derivation, &self.context)
            .map_err(|e| RemoteWorkerError::VerificationFailed(format!("{e:?}")))?;

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
        let root = kernel
            .prove_reflexivity(x.clone(), &mut Unbounded)
            .unwrap();
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

        let unknown_field = br#"{"worker_id":"worker-1","task_id":1,"result":{"Integer":[1,[1]]},"claim":{"NonZero":{"Integer":[1,[1]]}},"derivation":{"steps":[],"root":0},"worker_signature":[],"unexpected":true}"#;
        assert_eq!(
            RemoteCandidate::decode_json(unknown_field),
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
            RemoteWorkerError::VerificationFailed(
                RemoteVerificationFailure::ClaimDiscrepancy
            )
        );
        assert!(!error.to_string().contains("private_formula_name"));
    }
}
