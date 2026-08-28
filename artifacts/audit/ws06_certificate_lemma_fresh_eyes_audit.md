# WS06 CertificateLemma fresh-eyes audit

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-28T07:30Z
**Scope:** read-only diagnostic. No code edits. No new tests.
**Method:** mapped the WS06 reopen history (especially comment 25 at 2026-08-24T17:22:57Z, which named the `UnverifiedCertificateLemma` gap, and the latest OliveHawk comment noting "trusted certificate-family dispatch... still unearned") to file:line evidence at HEAD `542961a`.
**Source of truth:** `/data/projects/frankensympy/crates/fsym-proof-kernel/src/{kernel.rs,rule.rs,lib.rs,mutation.rs}`, `/data/projects/frankensympy/crates/fsym-runtime/src/remote_worker.rs`, plus downstream certificate producers in `fsym-polys`, `fsym-matrices`, and `fsym-core`.

---

## Summary verdict

The `ProofRule::CertificateLemma` rule is **structurally defined but unconditionally rejected**. The family-string on the rule is read for byte-budget accounting at `kernel.rs:692-694` and propagated by clone at `:1464-1472`, but `check_rule_application` at `kernel.rs:930-936` returns `Err(KernelError::UnverifiedCertificateLemma)` for **every** family string without dispatch. There is no family-dispatch table, no register-family API, and no per-family preconditions anywhere in `fsym-proof-kernel`.

**No family is currently safe to admit** without first defining three missing structural pieces:

1. A family-dispatch registry (currently does not exist)
2. A receipt-verification primitive (the `receipt_digest: [u8;32]` field on the rule is opaque and never checked)
3. Per-family preconditions (the rule carries a free-floating `Claim` and an assumption-context reference, but no family has registered what those mean)

The library docstring at `crates/fsym-proof-kernel/src/lib.rs:12` explicitly registers the gap: **"Fail-closed certificate lemma syntax pending a trusted family dispatcher"**.

---

## Gate-by-gate evidence

### Gate A — rejection locus and family-string handling

**Status:** REJECTION IS CORRECT (fail-closed). Family dispatch is the unearned gate.

**Reopen quote (comment 25 at 2026-08-24T17:22:57Z):**
> "The independent verifier accepts any deserialized CertificateLemma claim without certificate dispatch or receipt verification, and the broad simplify_normal_form / polynomial_ring_equivalence rules return arbitrary lhs=rhs claims without checking equivalence."

**Evidence:**

- `crates/fsym-proof-kernel/src/kernel.rs:930-936` — the rejection arm of `check_rule_application`:
  ```rust
  // (truncated; per audit, kernel.rs:930-936 unconditionally returns
  //  Err(KernelError::UnverifiedCertificateLemma) for any CertificateLemma step,
  //  irrespective of the family string, the claim, or the receipt_digest.)
  ```
  This is the single rejection site; every `ProofRule::CertificateLemma` step terminates here.

- `crates/fsym-proof-kernel/src/rule.rs:50-55` — the rule variant:
  ```rust
  CertificateLemma {
      family: String,
      claim: Claim,
      receipt_digest: [u8; 32],
  }
  ```
  Three fields. None of them are validated at construction time; `family` is a `String` carrier, not a typed identifier. (This is the same anti-pattern as the WS04 audit Gate 5 finding: magic strings as IR.)

- `crates/fsym-proof-kernel/src/kernel.rs:692-694` — preflight consumes the family string for byte-budget accounting only:
  ```rust
  // preflight counts the family-string bytes toward the derivation's
  // byte budget, but does not match the family against any registry.
  ```
- `crates/fsym-proof-kernel/src/kernel.rs:1464-1472` — clone propagates the family string (no validation).

- `crates/fsym-proof-kernel/src/lib.rs:12` — module docstring names the missing component: **"trusted family dispatcher"**.

- `crates/fsym-proof-kernel/src/mutation.rs:160-182` — negative test enforces the current fail-closed behavior. The test exercises rejection of `family="unregistered-forged-family"` with a `[0x42;32]` receipt. The test would need to be **re-shaped** to whitelist a real family, not deleted.

- `crates/fsym-runtime/src/remote_worker.rs:78` — `UnverifiedCertificateLemma` is mapped into a remote-worker error enum. The rejection propagates as a typed error all the way to remote workers, which is the correct fail-closed posture for the *unverified* state. Once a family is whitelisted, the error path remains the right behavior for non-whitelisted families.

### Gate B — downstream certificate producers (the unwired assets)

**Status:** Self-contained but unwired into the proof kernel.

**Evidence:**

- `crates/fsym-polys/src/gcd.rs:14-25` — `BezoutCertificate` struct (the GCD identity `au + bv = gcd(a,b)`). Paired with `verify_bezout_certificate` at `:149-188`. **Self-contained, unwired.**

- `crates/fsym-matrices/src/lib.rs` — four matrix-certificate producers (`CharpolyCertificate`, `LuCertificate`, `LinearSystemCertificate`, `QrCertificate`) and their `verify_*` functions. **Self-contained, unwired.** `MatrixError::UnsupportedCertificateDomain` at `:64-68` already exists, indicating the matrix side is prepared for "this certificate family isn't admitted" as a typed refusal.

- `crates/fsym-polys/src/groebner.rs:580-585` — `GroebnerBasisCertificate` plus `verify_groebner_certificate` at `:611-`. **Self-contained, unwired.**

- `crates/fsym-core/src/algebraic.rs:243-251` — `AlgebraicNumber` carries a certified root-isolating ball (Sturm-based). **Self-contained, unwired.**

- `crates/fsym-core/src/ball.rs:22-50` — Certified real ball `B(m, r)`. **No certificate envelope, no proof-kernel wiring.** This is the lowest-friction candidate to wire first, because it does not even need a producer-cert: it is a primitive that already exists.

### Gate C — receipts and preconditions (the missing structural pieces)

**Status:** Structural pieces do not exist; the rule carries them as opaque data only.

**Evidence:**

- The `receipt_digest: [u8; 32]` field on `ProofRule::CertificateLemma` is a 32-byte opaque blob. **No code path checks it against any expected hash.** The receipt is a published artifact, but the proof kernel never asks "is this receipt the receipt we expect for this family?".

- The `claim: Claim` field on the rule is a free-floating equality/algebraic identity. **The kernel does not check that the claim is the claim that the certificate family should produce.** For Bezout, the claim should be `au + bv = gcd(a,b)`; for LU, the claim should be `PA = LU`. Today the claim is whatever the producer says, with no family-specific validation.

- The assumption context is referenced in the `EvidenceClass::CertificateVerified` discharge semantics at `crates/fsym-outcome/src/lib.rs:98-110`, but the assumption context is **not threaded into the per-family preconditions** of the proof kernel. The `fsym-outcome` `CertificateVerified` class is loosely aware of contexts, but the kernel is not.

- `fsym-outcome/src/lib.rs:26-27` — the evidence class registry mentions `CertificateVerified` as a non-terminal class, meaning certificates never become terminal without further check. This is the correct posture; the gap is that the further check does not exist.

---

## What is needed (the three missing structural pieces)

For any family to be safely admitted, the proof kernel needs:

1. **A family-dispatch table** — a `HashMap<&'static str, FamilyHandler>` (or an `enum Family` with `dispatch` impl) registered at crate load time. The handler for family `X` is the only authority that decides whether a `CertificateLemma { family: "X", claim, receipt_digest }` step is accepted.

2. **A receipt-verification primitive** — a function `verify_receipt(receipt_digest, family) -> Result<Receipt, KernelError>` that knows the canonical receipt shape for each registered family. The 32-byte digest on the rule is the commitment to a specific receipt; today no code path makes that commitment verifiable.

3. **Per-family preconditions** — a function `preconditions(claim, context) -> Result<(), KernelError>` that checks, for each family, that the `Claim` carried on the rule is the right shape for that family under the supplied assumptions context. For Bezout: the claim must be `a*u + b*v = gcd(a,b)` with `u, v` in the right domain. For LU: the claim must be `P*A = L*U` with the right dimensions. Today no family has a precondition.

All three pieces need to be designed **before** the first family is admitted, because admitting a family without all three is exactly the laundering path the reopen comments name.

---

## The minimal-first-family recommendation

**`RealBall` (from `fsym-core/src/ball.rs:22-50`)** is the minimal-first family to admit, *if* the three structural pieces are designed first. Rationale:

- It is already a primitive — the `RealBall` is constructed by the `Ball` constructor and validated by `contains(value) -> bool`. There is no separate "producer" that needs to be wired in.
- The certificate is *self-evidencing*: a ball `B(m, r)` proves that the true value lies in `[m - r, m + r]`. The "claim" is `|x - m| <= r`; the "receipt" is the construction of `m` and `r` from a known algorithm.
- It is already fail-closed: the `RealBall` does not return a tight bound unless it can certify it.

The bounded hardening slice would be:

1. Add a `Family` enum (or `&'static str` whitelist) to `fsym-proof-kernel/src/kernel.rs`. For the first slice, only `RealBall` is admitted.
2. Add `dispatch_certificate_lemma(family, claim, receipt_digest, context) -> Result<(), KernelError>` that matches on family and delegates to a per-family handler.
3. Add a per-family handler for `RealBall` that (a) checks `claim` is an inequality `|expr - center| <= radius` shape, (b) checks `context` has the numeric-domain preconditions, (c) checks `receipt_digest` is the canonical digest of the `RealBall` construction.
4. Replace the unconditional `UnverifiedCertificateLemma` arm at `kernel.rs:930-936` with a dispatch that calls the new function. **Keep** the `UnverifiedCertificateLemma` arm for unknown families — that is still the correct refusal.
5. Re-shape the negative test at `mutation.rs:160-182` to verify both: rejection of an unknown family (already present) AND acceptance of the one whitelisted `RealBall` family (new).

**Do not** attempt to admit `BezoutCertificate`, `LuCertificate`, or `GroebnerBasisCertificate` in the same slice. Those need their own producer/verifier wiring and their own receipt shapes; the audit does not have evidence that the existing `verify_*` functions in `fsym-polys` and `fsym-matrices` produce the same `receipt_digest` shape the kernel would expect.

---

## Cross-cutting observations

### A. The `family: String` carrier should become a typed enum

`fsym-proof-kernel/src/rule.rs:50-55` defines `family` as `String`. This is the same anti-pattern as the WS04 audit Gate 5 (magic strings as IR). A bounded slice could either:
- Define a `CertificateFamily` enum with `as_str(&self) -> &'static str` and a `FromStr` that fails closed on unknown family names, or
- Use a `&'static str` whitelist and a compile-time check that all admitted families are listed.

The second is cheaper; the first is more type-safe.

### B. The `receipt_digest: [u8; 32]` should be a typed ID

`fsym-id/src/lib.rs` already defines `ReceiptId` via `define_id!` (line 261-265, prefix `"receipt"`). The `receipt_digest: [u8; 32]` on the rule could be replaced with `ReceiptId` (the typed 64-bit truncation) or with the full 32-byte BLAKE3 digest if a wider ID is added. The current raw `[u8; 32]` bypasses the typed-ID substrate entirely.

### C. The `Claim` on the rule should be family-specific

Today the `claim: Claim` is a generic `Claim` type. For each family, the claim should be a typed sub-enum:
- `BezoutClaim { a, b, u, v, gcd }`
- `LuClaim { p, a, l, u }`
- `RealBallClaim { expr, center, radius }`
- ...

This is a larger refactor; it should not be in the first slice. But the audit recommends it as the long-term design.

---

## File:line evidence index

| File | Lines | What is there | Gate |
|------|------|---------------|------|
| `crates/fsym-proof-kernel/src/kernel.rs` | 700-936 | `check_rule_application` — the single rule-by-rule authority | A |
| `crates/fsym-proof-kernel/src/kernel.rs` | 930-936 | Unconditional `UnverifiedCertificateLemma` rejection | A (rejection is correct; dispatch is the gap) |
| `crates/fsym-proof-kernel/src/kernel.rs` | 692-694 | Preflight byte-budget accounting consumes family string | A (no dispatch) |
| `crates/fsym-proof-kernel/src/kernel.rs` | 1464-1472 | Clone propagates family string (no validation) | A |
| `crates/fsym-proof-kernel/src/rule.rs` | 50-55 | `CertificateLemma { family, claim, receipt_digest }` definition | A, B, C (no validation) |
| `crates/fsym-proof-kernel/src/lib.rs` | 12 | Module docstring registers the missing dispatcher | A |
| `crates/fsym-proof-kernel/src/mutation.rs` | 160-182 | Negative test enforces current fail-closed behavior | A (test would need re-shaping to whitelist a real family) |
| `crates/fsym-runtime/src/remote_worker.rs` | 78 | Maps `UnverifiedCertificateLemma` into remote-worker error | A (correct propagation) |
| `crates/fsym-polys/src/gcd.rs` | 14-25, 149-188 | `BezoutCertificate` + `verify_bezout_certificate` (self-contained, unwired) | B |
| `crates/fsym-matrices/src/lib.rs` | 64-68 | `MatrixError::UnsupportedCertificateDomain` (typed refusal exists) | B |
| `crates/fsym-matrices/src/lib.rs` | (various) | `CharpolyCertificate`, `LuCertificate`, `LinearSystemCertificate`, `QrCertificate` + their `verify_*` functions (self-contained, unwired) | B |
| `crates/fsym-polys/src/groebner.rs` | 580-585, 611+ | `GroebnerBasisCertificate` + `verify_groebner_certificate` (self-contained, unwired) | B |
| `crates/fsym-core/src/algebraic.rs` | 243-251 | `AlgebraicNumber` certified root-isolating ball (self-contained, unwired) | B |
| `crates/fsym-core/src/ball.rs` | 22-50 | `RealBall B(m, r)` (primitive, no envelope, no proof-kernel wiring) | B (minimal-first candidate) |
| `crates/fsym-outcome/src/lib.rs` | 26-27, 98-110 | `CertificateVerified` evidence class registry and discharge semantics | C |

---

## Recommendation to the WS06 owner (OliveHawk)

The smallest responsible bounded slice to close **one** family is:

1. **Add the three structural pieces** (family-dispatch table, receipt-verification primitive, per-family preconditions) as a single ~100-200 line commit. **Do not** populate any family yet — leave the dispatch empty except for a single family.
2. **Whitelist `RealBall` from `fsym-core/src/ball.rs:22-50`** as the first admitted family. The handler checks that `claim` is an inequality shape and that `receipt_digest` matches the canonical `RealBall` digest.
3. **Re-shape the mutation test at `mutation.rs:160-182`** to verify both: rejection of an unknown family (existing behavior) AND acceptance of the whitelisted `RealBall` family with a valid `RealBall` construction.
4. **Keep all other downstream certificates unwired** for now. They each need their own slice; mixing them in the same commit is a `conformance metastasis` anti-pattern (§5 of AGENTS.md).

**Do not** attempt to wire `BezoutCertificate`, `LuCertificate`, `GroebnerBasisCertificate`, or `CharpolyCertificate` in the same slice. Each of those has its own producer-verifier pairing and its own receipt shape; the audit cannot confirm they produce the canonical digest the kernel would expect without re-reading the producer code in detail (which is out of scope for this diagnostic).

**Do not** change the `family: String` field on the rule in this slice. A typed `CertificateFamily` enum is a follow-up hardening that should land after the first family is admitted, so the diff is small and reviewable.

---

## Honesty note

This audit was produced by the DustyAspen subagent. The file:line evidence was read from the current working tree at HEAD `542961a`; no `cargo` invocation was performed. The audit does not exhaustively read the downstream `verify_*` functions in `fsym-polys/src/gcd.rs:149-188` or `fsym-matrices/src/lib.rs`; the `BezoutCertificate` / `LuCertificate` / etc. claims are based on the audit subagent's structural read, not on a line-by-line verification of the verification code. If a downstream `verify_*` function has bugs, this audit would not catch them.

The audit is intended as a diagnostic input for the WS06 owner (OliveHawk). It does not claim WS06 closure, does not propose implementation, and does not pre-allocate work to any agent. The "minimal-first family" recommendation (`RealBall`) is based on the structural simplicity of the `fsym-core/src/ball.rs` primitive, not on a runtime comparison of which family is *most needed* for downstream users. The WS06 owner may prefer a different first family based on application priority; the audit only describes the structural ease of admission.
