# WS06 Proof Kernel and Evidence System — Acceptance & Mutation Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws06-proof-kernel-9j1`
**Audit Scope:** Verification of the derivation-DAG proof kernel (`fsym-proof-kernel`), evidence envelope and promotion lattice (`fsym-evidence`), certificate lemmas, and registered weakening mutants.

---

## 1. Executive Summary & Acceptance Evidence

All proof kernel and evidence system criteria were verified:

### Criterion 1: Zero Test Failures Across Proof Kernel & Evidence Crates
```bash
cargo test -p fsym-proof-kernel -p fsym-evidence
```
**Outcome:** **55 tests passed, 0 failed, 0 ignored**:
- `fsym-proof-kernel`: 32 unit & mutation tests passed
- `fsym-evidence`: 23 lattice, namespace, receipt, and envelope integrity tests passed

### Criterion 2: Registered Weakening Mutants Killed for All Certificate Families
Every certificate family claimed in code (`RealBall` and `Opaque`) has registered weakening mutants killed in `crates/fsym-proof-kernel/src/mutation.rs`:
- `mutant_unchecked_certificate_lemma_killed`: Unchecked/unregistered families rejected fail-closed with `KernelError::UnverifiedCertificateLemma`.
- `mutant_opaque_payload_for_real_ball_rejected`: Attempting to smuggle an `Opaque` receipt digest as a `RealBall` certificate is rejected fail-closed (`KernelError::InvalidCertificateLemma`).
- `mutant_inconsistent_real_ball_claims_rejected`: Invalid/contradictory sign assertions on real balls rejected fail-closed.
- `mutant_real_ball_zero_to_zero_admission_killed`: Forged `NonZero(0^0)` claims rejected fail-closed when base contains zero.
- `mutant_real_ball_enclosure_check_deletion_killed`: Deletion of expression-enclosure containment check is caught and killed.
- `real_ball_refuses_equality_without_equality_specific_payload`: Equality claims rejected without equality-specific proof payload.
- `registered_family_cannot_authorize_unrelated_symbol`: Unrelated symbols rejected.
- `oversized_real_ball_certificate_is_refused_by_preflight`: Preflight accounting refuses oversized certificate payloads before arithmetic allocation.
- `real_ball_power_growth_is_refused_before_oversized_multiplication`: Intermediate limb growth limits enforced before allocation.

In addition, algebraic and definitional reduction mutants are killed:
- `mutant_broad_normal_form_claim_killed`
- `mutant_partial_values_cannot_be_erased_by_polynomial_zero`
- `mutant_polynomial_proof_refuses_excessive_depth_at_trust_boundary`
- `mutant_claim_tampering_in_derivation_tree_killed`
- `mutant_transitivity_mismatch_killed`
- `mutant_forward_or_self_reference_killed`
- `mutant_forged_definitional_arithmetic_killed`
- `mutant_forged_context_predicate_killed`
- `pythagorean_identity_proves_folded_pairs_and_kills_mutants`

---

## 2. Invariants & Architecture

1. **Independent Reference Lane:**
   `verify_derivation_independent` independently walks exported `DerivationTree`s and verifies every inference step without referencing generator state.
2. **Evidence Promotion Lattice (`fsym-evidence`):**
   Illegal conversions (e.g., `HeuristicCandidate -> KernelProved`, `OracleConformant -> KernelProved`, `CertifiedNumeric -> KernelProved`) strictly fail closed via `validate_evidence_transition`.
3. **Separate Namespaces:**
   Candidate namespace and verified namespace remain physically separate (`CandidateNamespace` vs `VerifiedNamespace`). Unverified derivations cannot enter verified storage.
4. **Resource Governance & Atomic Charges:**
   `DerivationPreflight` preflights step counts, expression depths, and limb bounds. Interrupted or cancelled derivations leave zero verified steps (`cancellation_before_publication_does_not_leave_a_verified_step`).
