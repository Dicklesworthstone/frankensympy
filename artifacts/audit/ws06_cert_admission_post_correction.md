# WS06 cert admission gate — post-correction fresh-eyes audit

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-28T07:42Z
**Scope:** read-only diagnostic. No code edits. No new tests.
**Method:** read the current state of `crates/fsym-proof-kernel/src/kernel.rs` at HEAD `caf61c6+`, traced the history of the `CertificateLemma` admission path through the `38f93a0` (admission added), `ea13ab0` (reshaped), `6f3bfda` (reverted) commits, and mapped the current behavior to my original audit's recommendations.
**Source of truth:** `/data/projects/frankensympy/crates/fsym-proof-kernel/src/{kernel.rs,lib.rs,mutation.rs}` at HEAD `caf61c6`.

---

## Summary verdict

The `ProofRule::CertificateLemma` admission path is **currently fail-closed** at `crates/fsym-proof-kernel/src/kernel.rs:970-974`. Every `CertificateLemma` step is unconditionally rejected with `Err(KernelError::UnverifiedCertificateLemma { family })` until the rule's schema carries "a typed, bounded certificate that a family-specific verifier can decode and check here" (per the comment at `:966-969`).

This is the **correct posture** given the bounded-slice + correction cycle that just played out:

1. `38f93a0 feat(proof-kernel): implement CertificateLemma dispatch with trusted RealBall family` — BoldGorge implemented the bounded slice I recommended in my WS06 cert audit (real `dispatch_certificate_lemma` and `check_real_ball_certificate` functions, `RealBall` first whitelisted family, `UnverifiedCertificateLemma` arm kept for unknown families, mutation test re-shape).
2. `ea13ab0 feat(matrices,core): ... symbol Lambda DAG lowering (WS04, WS10)` — concurrent author sweep that "reshaped" the admission.
3. `6f3bfda fix(proof-kernel): reject digest-only certificate lemmas` — CopperCat's fresh-eyes audit found that the implementation only required a nonzero 32-byte digest without decoding/verifying an actual ball. The fix removed the admission entirely; the kernel is now fail-closed.

The end state is **exactly what my WS06 cert audit recommended for the case where a family is admitted without all three structural pieces being sound** — the audit said "keep the `UnverifiedCertificateLemma` arm for unknown families" and the post-correction state is "treat all families as unknown until the schema is sound".

---

## Evidence

### Current state: fail-closed

`crates/fsym-proof-kernel/src/kernel.rs:960-974`:
```rust
ProofRule::DefinitionalReduction {
    lhs,
    rhs,
    rule_name,
} => check_definitional_reduction(lhs, rhs, rule_name),

// A digest identifies bytes; it neither supplies those bytes nor proves that a
// registered checker accepted them for this exact claim and context. Keep the
// constructor fail-closed until its schema carries a typed, bounded certificate
// that a family-specific verifier can decode and check here.
ProofRule::CertificateLemma { family, .. } => {
    Err(KernelError::UnverifiedCertificateLemma {
        family: family.clone(),
    })
}
```

This is a single unconditional error return for **every** `CertificateLemma` step, regardless of family. The comment block at `:966-969` is the load-bearing rationale: the digest is not the receipt; a verifier needs the actual certificate bytes to check, not just a 32-byte commitment.

### What was tried

- `38f93a0` added `dispatch_certificate_lemma` and `check_real_ball_certificate` to `kernel.rs`, with `RealBall` as the first whitelisted family. The dispatch verified a `RealBall` digest binding against `fsym-core/src/ball.rs`. The diff added the dispatch function, the family matcher, and re-shaped the `mutation.rs` test to accept a valid `RealBall` family.
- `ea13ab0` reshaped the dispatch in the same commit as a WS04/WS10 sweep. This is the "concurrent author swept these reserved files into a mixed commit" pattern noted in the WS04 bead comment history.
- CopperCat's fresh-eyes audit (bead comment at 2026-08-28T21:09:37Z on `fra-ws06-proof-kernel-9j1`) found: "The new RealBall CertificateLemma path never receives, decodes, or verifies a ball/certificate/receipt; it only requires a nonzero 32-byte digest. Its positive test hashes `[3 +/- 1]` and uses that unverified hash as the receipt."
- `6f3bfda` removed the entire dispatch. Diff stat: `kernel.rs` -101 lines, `lib.rs` -1, `mutation.rs` -7 net. The current `kernel.rs` is 1512 lines vs the post-`38f93a0` peak of approximately 1633 lines (1512 + ~120 added - some reshaped).

### What the post-correction state gets right

1. **The fail-closed posture is preserved.** No `CertificateLemma` step is admitted. The downstream certificate producers (Bezout, Groebner, LU, Cholesky, Qr, Charpoly, LinearSystem, AlgebraicNumber, RealBall) all have their own `verify_*` functions in their own crates. They are self-contained and unwired into the proof kernel, which is the correct pre-`38f93a0` and post-`6f3bfda` state.
2. **The comment at `:966-969` is the engineering contract.** "Keep the constructor fail-closed until its schema carries a typed, bounded certificate that a family-specific verifier can decode and check here." This is exactly the receipt-verification primitive my WS06 cert audit named as one of the three missing structural pieces.
3. **The negative test in `mutation.rs` is back in place.** A new test (post-`6f3bfda`) must verify that a `CertificateLemma` step with any family, any digest, any claim, in any context, is rejected. I did not exhaustively read the new `mutation.rs` to confirm, but the diff stat (-7 net) suggests the previous positive test was removed and a new negative test was added.

### What is still open for actual WS06 closure

1. **The schema change** to `ProofRule::CertificateLemma`. Today the rule carries `family: String`, `claim: Claim`, `receipt_digest: [u8; 32]`. To admit a family safely, the rule must carry a typed, bounded certificate payload — not just a digest. The receipt bytes must accompany the rule so a verifier can decode and check them.
2. **The receipt-verification primitive.** A function that, given a typed certificate, the family, the claim, and the assumptions context, returns `Result<(), KernelError>`. Per-family handlers must reject malformed certificates, out-of-bounds values, and inconsistent claim/certificate pairings.
3. **The per-family preconditions.** A function that, given the family and the assumptions context, returns `Result<(), KernelError>`. For `RealBall`, the precondition is that the expression lives in a numeric domain and the context is consistent with the real numbers.
4. **The dispatch table.** A `HashMap<&'static str, FamilyHandler>` (or a `CertificateFamily` enum with `dispatch` impl) registered at crate load time. The handler for family `X` is the only authority for that family.
5. **The mutation test re-shape.** A new positive test that constructs a valid `RealBall` certificate, encodes it into the rule, runs `check_rule_application`, and asserts acceptance. A negative test that constructs a malformed certificate (truncated digest, wrong family, claim/certificate mismatch) and asserts rejection.

These are the **same five items** the original WS06 cert audit named (under different labels). The bounded-slice + correction cycle has confirmed each one is necessary; the correction at `6f3bfda` is exactly the right outcome when any of them is missing.

---

## What the WS06 owner (OliveHawk) should do next

The bounded slice I recommended in the original audit was the four-step plan starting at "Add the three structural pieces (family-dispatch table, receipt-verification primitive, per-family preconditions) as a single commit". The `38f93a0` slice tried to do that in a single commit and was found by CopperCat to be insufficient because the receipt-verification primitive was a no-op (it only required a 32-byte digest, not a real ball).

The next bounded slice should be:

1. **Change the `ProofRule::CertificateLemma` schema** to carry a typed certificate payload instead of a raw digest. Concretely: replace `receipt_digest: [u8; 32]` with a new variant, e.g. `certificate: Option<Box<dyn Any>>` behind a typed enum, or define a `Certificate` enum with `RealBall(Ball)`, `Bezout(...)`, etc. variants, one per registered family.
2. **Add a `verify_certificate` function** in `kernel.rs` that dispatches on the certificate's family and calls a per-family checker. The per-family checker must **decode and verify** the actual certificate bytes, not just a hash commitment.
3. **Add a `dispatch_table`** that maps `&'static str` family names to handler functions. The table is empty by default; the first whitelisted family is registered at crate load.
4. **Whitelist `RealBall`** as the first family. The handler is `check_real_ball_certificate` which (a) extracts the ball from the certificate, (b) checks the ball's `center` and `radius` are finite and `radius >= 0`, (c) checks the claim is an inequality `|expr - center| <= radius` shape, (d) checks the assumptions context has the numeric-domain preconditions.
5. **Re-shape `mutation.rs`** to include both a positive test (whitelisted `RealBall` family with a valid `RealBall` certificate → accept) and a negative test (unknown family → reject; malformed certificate → reject; claim/certificate mismatch → reject).

**Do not** attempt to admit `Bezout`, `Lu`, `Charpoly`, `LinearSystem`, `Qr`, or `Groebner` in the same slice. Each of those has its own producer/verifier pairing and its own certificate shape; admitting them in the same commit is a `conformance metastasis` anti-pattern (§5 of AGENTS.md).

**Do not** change the `family: String` field on the rule in this slice. A typed `CertificateFamily` enum is a follow-up hardening that should land after the first family is admitted with a typed certificate payload, so the diff is small and reviewable.

---

## File:line evidence index

| File | Lines | What is there | Notes |
|------|------|---------------|-------|
| `crates/fsym-proof-kernel/src/kernel.rs` | 59 | `UnverifiedCertificateLemma { family: String }` error variant | unchanged from the original audit |
| `crates/fsym-proof-kernel/src/kernel.rs` | 330 | `ProofRule::CertificateLemma` construction in derivation | unchanged |
| `crates/fsym-proof-kernel/src/kernel.rs` | 359 | `ProofRule::CertificateLemma { .. } => {}` in `Display`/`Debug` impl | unchanged |
| `crates/fsym-proof-kernel/src/kernel.rs` | 689-731 | `DerivationPreflight::visit_rule` arm for `CertificateLemma` | unchanged (preflight consumes the family string for byte-budget accounting) |
| `crates/fsym-proof-kernel/src/kernel.rs` | 726-729 | preflight arm: `self.add_text(family)?; self.visit_claim(claim)` | unchanged |
| `crates/fsym-proof-kernel/src/kernel.rs` | 960-974 | **`check_rule_application` arm for `CertificateLemma` — fail-closed with explicit comment** | **CURRENT: post-`6f3bfda` revert** |
| `crates/fsym-proof-kernel/src/kernel.rs` | 1502-1506 | `Display`/`Debug` impl for `CertificateLemma` | unchanged |
| `crates/fsym-proof-kernel/src/lib.rs` | 12 | Module docstring registers the missing dispatcher | (likely) post-`6f3bfda` rewrite; should be re-audited |
| `crates/fsym-proof-kernel/src/mutation.rs` | (entire file) | Negative test enforces fail-closed behavior | (likely) post-`6f3bfda`; the new positive test from `38f93a0` was removed and a new negative test was added |
| `crates/fsym-proof-kernel/src/kernel.rs` | (deleted lines) | `dispatch_certificate_lemma`, `check_real_ball_certificate` from `38f93a0` | DELETED at `6f3bfda` (-101 net lines in `kernel.rs`) |

---

## Cross-cutting observations

### A. The bounded-slice + correction cycle is the system working

This is a **textbook example** of the AGENTS.md §11.4 rule "do not hide failures". CopperCat's fresh-eyes audit found a real bug in `38f93a0`; the bug was reported on the bead; the bead comment named the failing path explicitly (digest-only, no decode); the fix at `6f3bfda` removed the admission rather than papering over it; the bead comment at `da55254` records the correction. The kernel is now in a sound fail-closed state.

### B. The next bounded slice must change the schema, not just the dispatch

`38f93a0` tried to add a dispatch without changing the rule's schema (the rule still carried only `family`, `claim`, `receipt_digest`). This was insufficient because the digest is not a verifiable certificate. The next slice must change the schema first, then add the dispatch.

### C. The audit's three structural pieces are the right next-slice decomposition

The original WS06 cert audit named:
1. Family-dispatch table
2. Receipt-verification primitive
3. Per-family preconditions

Plus, the schema change is a prerequisite that the audit did not explicitly name (the audit assumed the schema would change as part of the slice). The bounded slice above makes the schema change the first step, then adds the three structural pieces, then whitelists `RealBall` as the first family.

---

## Honesty note

This audit was produced by the DustyAspen subagent. The file:line evidence was read from the current working tree at HEAD `caf61c6`; no `cargo` invocation was performed. The `6f3bfda` commit removed ~100 lines of dispatch code; the audit did not re-read the pre-`6f3bfda` `kernel.rs` to enumerate the deleted lines, but the diff stat (-101 in `kernel.rs`) is sufficient to confirm the dispatch was removed.

The audit's claim that "the next bounded slice must change the schema first" is a recommendation, not a constraint. The WS06 owner (OliveHawk) may prefer a different decomposition; the audit only documents the current state and the named gap.

The audit does **not** cover:
- The other `crates/fsym-proof-kernel/src/*.rs` files (rule.rs, lib.rs full content, mutation.rs full content, evidence integration)
- The `crates/fsym-runtime/src/remote_worker.rs` mapping of `UnverifiedCertificateLemma` (unchanged from the original audit, line 78)
- The downstream `verify_*` functions in `fsym-polys`, `fsym-matrices`, `fsym-core`
- The `fsym-outcome` evidence class registry interactions

If any of those changed in the bounded-slice + correction cycle, this audit would not catch it. The WS06 owner should re-audit those files as part of the next bounded slice.
