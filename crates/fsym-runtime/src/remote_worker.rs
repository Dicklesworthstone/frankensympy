//! Untrusted remote worker execution lane and coordinator verification (WS16 / architecture §7.8).
//!
//! Rule (§7.5, §7.8): Remote workers only generate candidates. The coordinator MUST
//! independently verify any candidate derivation before accepting it. Worker signatures
//! or votes NEVER substitute for mathematical proof.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::Expr;
use fsym_proof_kernel::{Claim, DerivationTree, verify_derivation_independent};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RemoteWorkerError {
    #[error("Candidate derivation verification failed: {0}")]
    VerificationFailed(String),
    #[error("Untrusted worker claim forgery: claimed result does not match verified claim")]
    ClaimForgery,
    #[error("Worker timeout or communication fault")]
    WorkerFault,
    #[error("Payload schema or integrity corruption")]
    CorruptedPayload,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedAcceptedResult {
    pub task_id: u64,
    pub result: Expr,
    pub claim: Claim,
    pub verifier_receipt_digest: [u8; 32],
}

/// Coordinator supervising untrusted remote workers.
pub struct CoordinatorVerifier {
    pub context: Arc<ImmutableAssumptionsSnapshot>,
}

impl CoordinatorVerifier {
    pub fn new(context: Arc<ImmutableAssumptionsSnapshot>) -> Self {
        Self { context }
    }

    /// Evaluates and verifies a candidate from an untrusted remote worker.
    ///
    /// Fails closed if the derivation is invalid, context predicates are unproven,
    /// or the claim is forged.
    pub fn verify_remote_candidate(
        &self,
        candidate: &RemoteCandidate,
    ) -> Result<VerifiedAcceptedResult, RemoteWorkerError> {
        // 1. Independent stateless verification of the derivation tree
        let verified_claim = verify_derivation_independent(&candidate.derivation, &self.context)
            .map_err(|e| RemoteWorkerError::VerificationFailed(format!("{e:?}")))?;

        // 2. Bind both the worker's claim and returned value to the exact verified root.
        // Derivation export order is not itself a statement about which step is the root.
        if verified_claim != candidate.claim || claimed_result(&verified_claim) != &candidate.result
        {
            return Err(RemoteWorkerError::ClaimForgery);
        }

        // 3. Compute BLAKE3 verifier receipt
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
        let verifier_receipt_digest = *hasher.finalize().as_bytes();

        Ok(VerifiedAcceptedResult {
            task_id: candidate.task_id,
            result: candidate.result.clone(),
            claim: candidate.claim.clone(),
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
        let coordinator = CoordinatorVerifier::new(context);
        let x = Expr::symbol("x");
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
}
