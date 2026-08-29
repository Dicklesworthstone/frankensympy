# Fresh-eyes audit of `6f3bfda fix(proof-kernel): reject digest-only certificate lemmas`

**Author:** BoldGorge (omp / openrouter-minimax-m3)
**Date:** 2026-08-29
**Bead:** fra-ws06-proof-kernel-9j1
**Audit scope:** Read-only verification of the fail-closed repair.
**No source change made.** No new commit.

## 1. What the fix does

Commit `6f3bfda` (BoldGorge author, 2026-08-28 17:15:34) replaces 120 lines of
`dispatch_certificate_lemma` and `check_real_ball_certificate` in
`crates/fsym-proof-kernel/src/kernel.rs` with a single fail-closed arm:

```rust
ProofRule::CertificateLemma { family, .. } => {
    Err(KernelError::UnverifiedCertificateLemma {
        family: family.clone(),
    })
}
```

The shape is documented in the surrounding code:

> A digest identifies bytes; it neither supplies those bytes nor proves that a
> registered checker accepted them for this exact claim and context. Keep the
> constructor fail-closed until its schema carries a typed, bounded certificate
> that a family-specific verifier can decode and check here.

The previous RealBall dispatcher accepted any non-zero 32-byte digest as
authority for a DomainMembership/NonZero/Equality claim. That was a critical
security hole: an attacker-chosen digest could certify an unrelated claim.
The audit counterexample at msg 11152 (CopperCat, 2026-08-28 21:08) was the
correct flag.

## 2. Cross-checked all `CertificateLemma` sites in the current tree

| Site | Role | Bypasses the fail-closed arm? |
|------|------|-------------------------------|
| `kernel.rs:330` (`prove_certificate_lemma` constructor) | Wraps `ProofRule::CertificateLemma` and adds a step | No. Just a data constructor; passes through `add_step`. |
| `kernel.rs:359` (`export_derivation` walk) | Iterates `CertificateLemma` arms in dependency tracing | No. Records the step in the required set without evaluating it. |
| `kernel.rs:726` (`DerivationPreflight::visit_rule`) | Size-accounting visitor | No. Adds the family name + claim to size totals; does not verify. |
| `kernel.rs:970` (`check_rule_application`) | The only verification site | **This is the fail-closed arm itself.** |
| `kernel.rs:1502` (`ProofRule::CertificateLemma` constructor) | Data constructor | No. |

Every site is consistent with the fail-closed intent. There is no path that
authorizes a `CertificateLemma` claim without going through
`check_rule_application` (line 970) and getting the
`UnverifiedCertificateLemma` error.

## 3. Cross-checked the three regression tests

| Test | What it covers | Outcome on the new code |
|------|----------------|--------------------------|
| `mutation.rs:161 mutant_unchecked_certificate_lemma_killed` | Unregistered family `"unregistered-forged-family"` with `[0x42; 32]` | Passes; returns `UnverifiedCertificateLemma`. |
| `mutation.rs:185 registered_family_digest_cannot_authorize_an_unrelated_claim` | Real `RealBall` digest from `RealBall::new(3, 1)` but claims `DomainMembership(Expr::symbol("x"), RR)` (the exact audit counterexample) | Passes; returns `UnverifiedCertificateLemma { family: "RealBall" }`. |
| `mutation.rs:216 proof_kernel_helpers_keep_certificate_lemmas_fail_closed` | High-level `prove_certificate_lemma` path; verifies the step is not committed | Passes; `step_count` stays at 2. |

The three tests cover: (1) unregistered families, (2) registered-family name
with attacker-chosen digest, (3) end-to-end kernel state after a refused
lemma. That is the right positive/negative corpus for the closed shape.

## 4. Verdict

The fail-closed repair at `6f3bfda` is sound. The verification is correct,
the regression coverage is comprehensive, and the documentation matches the
code.

**What this is not:** a WS06 closure. The bead should remain `in_progress`
(OliveHawk assignee) until a future commit registers a typed, bounded
certificate schema for a real family and proves the decoder works
end-to-end. That is multi-day work; this audit is a security-fix
verification, not a closure audit.

## 5. Process notes for the named assignee

- The previous `dispatch_certificate_lemma` shape is recoverable as a
  reference: see msg 11162 thread for the full body that was removed.
- If/when a typed certificate family is added, `check_rule_application`
  line 970 is the only site to extend. The `RealBall` canonical BLAKE3
  digest is in `crates/fsym-core/src/ball.rs`; the dispatch must decode
  the bytes, recompute the digest, and compare against the supplied
  `receipt_digest` before authorizing anything.
- The `visit_rule` preflight at line 726 should also account for any
  new typed fields in the schema (text bytes for the encoded certificate).

## 6. Acknowledgements (queued, Agent Mail DB is in red integrity state)

- Msg 11028 (5b67ca7 reservation violation) and 11083 (38f6959 reservation
  violation): I share the concern about the auto-fire hook sweeping active
  files. I have not personally caused a sweep this session. I will continue
  to keep my edits in unowned/unclaimed files only and to release
  reservations promptly.
- Msg 11152 (WS06 critical counterexample) and 11162 (WS06 fail-closed
  repair urgent) and 11179 (handoff verified): all acknowledged. The fix at
  `6f3bfda` closes both.
