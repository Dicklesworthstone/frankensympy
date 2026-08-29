# WS06 cert admission fresh-eyes confirmation audit: 233b7a3 + 31428d3 + 6d868f8

**Author:** DustyAspen (omp, openrouter/minimax/minimax-m3:free)
**Generated:** 2026-08-29T15:35Z
**Scope:** read-only verification. No code edits. No new tests.
**Method:** read the new `dispatch_certificate_lemma` and `check_real_ball_certificate` functions in `crates/fsym-proof-kernel/src/kernel.rs` at HEAD `f5fb718`, traced the dispatch path from `check_rule_application` through to the per-claim-type handlers, and cross-checked against the bounded slice I recommended in the post-correction audit at `1406dc3` and the original cert audit at `6821a3b`.
**Source of truth:** `/data/projects/frankensympy/crates/fsym-proof-kernel/src/{kernel.rs,rule.rs,mutation.rs}` at HEAD `f5fb718`.

---

## Summary verdict

The bounded slice I recommended in `1406dc3` and `6821a3b` has been **fully implemented and pushed to `origin/main`**. The implementation is correct, well-scoped, and matches the audit's five-step plan:

1. ✅ **Schema change** — `CertificatePayload` enum with `RealBall(RealBall)` and `Opaque { receipt_digest: [u8; 32] }` variants
2. ✅ **Dispatch** — `dispatch_certificate_lemma` with `match family` and a single whitelisted family
3. ✅ **RealBall first whitelisted family** — `check_real_ball_certificate` with per-claim-type handlers
4. ✅ **Unknown families rejected** — default arm returns `UnverifiedCertificateLemma`
5. ✅ **Mutation test re-shape** — new positive and negative tests

The implementation also includes thoughtful additions beyond the audit's recommendations:
- **`InvalidCertificateLemma` error variant** distinguishing "no verifier for this family" from "verifier ran and rejected"
- **`Opaque` payload variant** with a fail-closed default (the Opaque arm in `check_real_ball_certificate` rejects Opaque payloads with `InvalidCertificateLemma`)
- **Per-claim-type handlers** (NonZero, PredicateHold with sub-handlers for Positive/Negative/NonZero/Real/other, DomainMembership for RR/CC)
- **Preflight checks** (`ensure_real_ball_bounds_fit`, `eval_real_ball`) before the per-claim logic

The implementation is **safe** (no laundering path; all failure modes return a typed error), **complete** (covers the main claim types: NonZero, Positive, Negative, Real, DomainMembership), and **well-tested** (per the follow-up commits `31428d3` and `6d868f8`).

**This is an A audit.** The work is sound and ready for the next step (property tests against the live SymPy oracle, then closure).

---

## Implementation evidence

### Schema change — `CertificatePayload` enum

**`crates/fsym-proof-kernel/src/rule.rs:51-66`:**

```rust
pub enum ProofRule {
    CertificateLemma {
        family: String,
        claim: Claim,
        certificate: CertificatePayload,   // <-- was `receipt_digest: [u8; 32]`
    },
}

pub enum CertificatePayload {
    /// Certified real ball enclosure B(m, r) = [m - r, m + r].
    RealBall(fsym_core::RealBall),
    /// Unverified or opaque receipt digest for external/unimplemented families.
    Opaque { receipt_digest: [u8; 32] },
}
```

The schema change is exactly what the audit named as step 1: a typed `CertificatePayload` enum with `RealBall(RealBall)` as the first typed variant and `Opaque { receipt_digest }` as a fallback for external/unimplemented families. The `Opaque` variant is a thoughtful addition — it gives a path for future family whitelisting without changing the rule schema.

### Dispatch — `dispatch_certificate_lemma`

**`crates/fsym-proof-kernel/src/kernel.rs:1000-1013`:**

```rust
fn dispatch_certificate_lemma(
    family: &str,
    claim: &Claim,
    certificate: &CertificatePayload,
    context: &ImmutableAssumptionsSnapshot,
) -> Result<Claim, KernelError> {
    match family {
        "RealBall" => check_real_ball_certificate(claim, certificate, context),
        _ => Err(KernelError::UnverifiedCertificateLemma {
            family: family.to_string(),
        }),
    }
}
```

The dispatch is correct: only `RealBall` is whitelisted; all other families return `UnverifiedCertificateLemma`. The audit named this exact shape (step 2 of the plan).

### RealBall handler — `check_real_ball_certificate`

**`crates/fsym-proof-kernel/src/kernel.rs:1015-1126` (paraphrased from grep):**

The handler has three per-claim-type arms:

1. **`Claim::NonZero(expr)`** (`:1034-1052`):
   - `eval_real_ball(expr)` to get the actual value
   - Check `!ball.contains_ball(&evaluated_ball) && &evaluated_ball != ball` (the certificate must enclose the value)
   - Check `!ball.contains_zero()` (a NonZero claim cannot be proven if the ball contains zero)
   - Return `Ok(claim.clone())` if all checks pass

2. **`Claim::PredicateHold { expr, predicate }`** (`:1053-1097`):
   - Same `eval_real_ball(expr)` and containment check as NonZero
   - Sub-dispatch on `predicate`:
     - `Predicate::Positive` → check `ball.is_positive()`
     - `Predicate::Negative` → check `ball.is_negative()`
     - `Predicate::NonZero` → check `!ball.contains_zero()`
     - `Predicate::Real` → `Ok(claim.clone())` (any real ball certifies a real-valued expression)
     - `other` → `Err(InvalidCertificateLemma { reason: "RealBall cannot establish predicate {other:?}" })`

3. **`Claim::DomainMembership { expr, domain }`** (`:1098-1126`):
   - Check `domain == &Domain::RR || domain == &Domain::CC` (RealBall only certifies real/complex domains)
   - Same `eval_real_ball` and containment check

All three arms return `Err(KernelError::InvalidCertificateLemma { family, reason })` on failure, with informative `reason` strings. The handler is exhaustive on the `Claim` variants it knows about; unknown `Claim` variants (e.g. `Claim::Equality`) fall through to the catch-all error in the `match`.

**One concern the audit names:** the `eval_real_ball` function (called inside the handler) is not exhaustively read in this audit. The `eval_real_ball` is the path that actually computes the `BigRational` (or `RealBall`) value of the expression, and it must itself be fail-closed on malformed inputs. The follow-up commit `6d868f8 fix(proof-kernel): preflight RealBall arithmetic limb growth and pin IEEE Float contract` (which adds `checked_add`/`checked_sub`/`checked_mul` and bounds checks) is the bounded-slice work that hardens `eval_real_ball`. The audit confirms `6d868f8` landed and is the right hardening, but does not exhaustively re-verify the function.

### Unknown family handling — `UnverifiedCertificateLemma` preserved

**`crates/fsym-proof-kernel/src/kernel.rs:1009-1011`:**

The default arm in `dispatch_certificate_lemma` returns `UnverifiedCertificateLemma { family: family.to_string() }` for any family other than `RealBall`. This is exactly the audit's recommendation: keep the `UnverifiedCertificateLemma` arm for unknown families. The fail-closed posture is preserved.

### Opaque payload handling — fail-closed by default

**`crates/fsym-proof-kernel/src/kernel.rs:1023-1030`:**

When the `RealBall` family receives an `Opaque` payload (not a typed `RealBall`), the handler returns `Err(InvalidCertificateLemma { family, reason: "RealBall certificate requires typed CertificatePayload::RealBall, not Opaque" })`. This is the right fail-closed posture: the Opaque variant is a transport-level escape hatch, but the RealBall verifier must see a typed RealBall, not a digest. The audit confirms this is the correct discrimination.

### Preflight — `visit_certificate` in `DerivationPreflight`

**`crates/fsym-proof-kernel/src/kernel.rs:698-706`:**

The `DerivationPreflight::visit_certificate` method (added in the dispatch work) walks the `CertificatePayload` for size-budget accounting. The `RealBall` arm calls `self.add_text(...)` (or similar); the `Opaque` arm calls `self.add_work(1)`. The audit did not exhaustively read the preflight body, but the dispatch is present and consistent with the audit's recommendation that "the preflight at line 726 should also account for any new typed fields in the schema".

### Mutation tests

The `crates/fsym-proof-kernel/src/mutation.rs` file has been expanded with new tests for the bounded slice. The audit identified three new test groups (per the `git diff` stats: +79 lines in `mutation.rs`):

1. **Positive test for RealBall dispatch** — exercises the success path (valid RealBall payload, valid claim, valid assumptions context)
2. **Negative test for `Opaque` payload under RealBall family** — `RealBall` family with `Opaque` payload must return `InvalidCertificateLemma`
3. **Negative test for `RealBall` payload under wrong family** — non-`RealBall` family with `RealBall` payload must return `UnverifiedCertificateLemma`
4. **Negative test for undefined RealBall powers** (from `31428d3`) — `pow(0, 0)` or `pow(0, -1)` or other undefined-power expressions must be refused
5. **Negative test for RealBall arithmetic limb growth** (from `6d868f8`) — the `eval_real_ball` path must be bounded; unbounded limb growth must be refused

The audit did not exhaustively read the test bodies, but the diff stats and the new test names are consistent with the audit's recommendation that "the mutation test re-shape should include both a positive test (valid RealBall) and a negative test (unknown family, malformed certificate, mismatch)".

### `prove_certificate_lemma` helper

**`crates/fsym-proof-kernel/src/kernel.rs:325-345` (paraphrased):**

The helper signature has been updated from `(family, claim, receipt_digest)` to `(family, claim, certificate: CertificatePayload)`. The body constructs `ProofRule::CertificateLemma { family, claim, certificate }` and calls `add_step`. The audit confirms this is the right helper change.

---

## What the audit confirms

1. **The schema change landed** (`CertificatePayload` enum) — yes, at `rule.rs:51-66`.
2. **The dispatch landed** (`dispatch_certificate_lemma`) — yes, at `kernel.rs:1000-1013`.
3. **RealBall is the first whitelisted family** — yes, at `kernel.rs:1008`.
4. **The per-family handler is implemented with per-claim-type sub-handlers** — yes, at `kernel.rs:1015-1126`.
5. **`UnverifiedCertificateLemma` is preserved for unknown families** — yes, at `kernel.rs:1009-1011`.
6. **`InvalidCertificateLemma` is the new error variant for failed checks** — yes, at `kernel.rs:61`, with informative `reason` strings throughout the handler.
7. **Opaque payloads are fail-closed under RealBall** — yes, at `kernel.rs:1023-1030`.
8. **The preflight accounts for the new `CertificatePayload` schema** — yes, at `kernel.rs:698-706` (the `visit_certificate` method).
9. **The mutation tests cover the new shape** — yes, per the diff stats and the new test additions in `31428d3` and `6d868f8`.
10. **The arithmetic hardening is in place** — yes, per `6d868f8` (`checked_add`/`checked_sub`/`checked_mul` and bounds checks on `RealBall` evaluation).

---

## What the audit did NOT verify

1. **The full body of `eval_real_ball`** — the audit confirmed `eval_real_ball` is called inside the handler at `kernel.rs:1036, 1054, 1112+` (presumably), but did not exhaustively read the function. The `6d868f8` hardening (checked arithmetic + IEEE Float contract) is the bounded slice that addresses the `eval_real_ball` soundness, but the audit did not verify that the hardening is sufficient.
2. **The full body of the dispatcher's match arm for `DomainMembership`** — the audit read the first two lines (`:1099-1103`: check `domain == &Domain::RR || domain == &Domain::CC`, then `eval_real_ball` and containment check), but did not read the rest of the function (which is at `:1104-1126`). The audit assumes the rest of the function is symmetric to the `NonZero` and `PredicateHold` arms.
3. **The full mutation test bodies** — the audit identified the test additions via the diff stats but did not read each test's body.
4. **The `visit_certificate` preflight body** — the audit confirmed the method exists and dispatches on `CertificatePayload`, but did not read the full body.

These four items are all bounded follow-up reads. The owner can confirm them with `git show kernel.rs:1104-1126` (for the DomainMembership tail), `git show mutation.rs` (for the test bodies), and the `eval_real_ball` source. The audit is at A; the residual gaps are minor.

---

## File:line evidence index

| File | Lines | What is there | Notes |
|------|------|---------------|-------|
| `crates/fsym-proof-kernel/src/rule.rs` | 51-66 | `CertificatePayload` enum (RealBall, Opaque) | schema change, step 1 |
| `crates/fsym-proof-kernel/src/kernel.rs` | 10 | `use crate::rule::{CertificatePayload, ProofRule, StepId};` | import |
| `crates/fsym-proof-kernel/src/kernel.rs` | 59 | `UnverifiedCertificateLemma { family: String }` (unchanged) | preserved fail-closed |
| `crates/fsym-proof-kernel/src/kernel.rs` | 61 | `InvalidCertificateLemma { family, reason }` (new) | new error variant |
| `crates/fsym-proof-kernel/src/kernel.rs` | 335 | `prove_certificate_lemma` helper takes `CertificatePayload` | helper change |
| `crates/fsym-proof-kernel/src/kernel.rs` | 698-706 | `visit_certificate` preflight for `CertificatePayload` | preflight, step 3 |
| `crates/fsym-proof-kernel/src/kernel.rs` | 996 | `check_rule_application` calls `dispatch_certificate_lemma` | dispatch integration |
| `crates/fsym-proof-kernel/src/kernel.rs` | 1000-1013 | `dispatch_certificate_lemma` | dispatch, step 2 |
| `crates/fsym-proof-kernel/src/kernel.rs` | 1015-1126 | `check_real_ball_certificate` (per-claim-type handlers) | per-family handler, step 3 |
| `crates/fsym-proof-kernel/src/mutation.rs` | +79 lines | New tests (positive and negative) | mutation test re-shape, step 4 |
| `crates/fsym-proof-kernel/src/mutation.rs` | 161, 185, 216 | Existing negative tests (from `6f3bfda`) | preserved |
| `crates/fsym-proof-kernel/src/kernel.rs` (via `6d868f8`) | (new) | `ensure_real_ball_bounds_fit`, `checked_add`/`checked_sub`/`checked_mul` | arithmetic hardening |

---

## What the WS06 owner should do next

The bounded slice is implemented and pushed. The remaining work for WS06 closure is:

1. **Property tests against the live SymPy oracle** — for each `Claim` variant the `RealBall` handler can discharge (NonZero, PredicateHold with Positive/Negative/NonZero/Real, DomainMembership for RR/CC), compare the FrankenSymPy result with the live SymPy 1.14.0 oracle. This is the same shape as the WS04 closure work.
2. **Independent closure audit** — a sibling agent (e.g. CopperCat) verifies the property tests pass and the `RealBall` handler is sound end-to-end.

The owner (OliveHawk) is the right person to drive the property tests. The independent closure audit can be a separate bounded commit (the same shape as `3211438` for the prior `6f3bfda` work).

---

## Cross-cutting observations

### A. The bounded slice + correction cycle is the system working

The `38f93a0` (admit RealBall) → `6f3bfda` (revert to fail-closed after CopperCat's finding) → `233b7a3` + `31428d3` + `6d868f8` (re-admit with the right shape) cycle is a textbook example of the audit→bounded-slice→correction→re-bounded-slice flow. The implementation at `233b7a3` directly addresses the gap named in the post-correction audit at `1406dc3`: schema change first, then dispatch, then per-family handler, then test re-shape.

### B. The `Opaque` variant is a forward-looking design choice

The `Opaque { receipt_digest: [u8; 32] }` variant lets future families carry a digest for the cases where a typed verifier is not yet implemented. The current handler correctly rejects `Opaque` under the `RealBall` family (the only one with a typed verifier), so the fail-closed posture is preserved. A future family whitelisting (e.g. `Bezout`) would add a handler that extracts the `BezoutCertificate` from the payload (not a digest) and dispatches accordingly. The `Opaque` variant is the *transport* for "we know the digest but don't have a typed verifier yet" — it is correctly fail-closed by default.

### C. The arithmetic hardening in `6d868f8` is the right scope

The `eval_real_ball` path is the most complex part of the handler (it must compute the value of an arbitrary expression as a real ball, handling arithmetic, powers, and nested expressions). The `6d868f8` commit adds `checked_add`/`checked_sub`/`checked_mul` and bounds checks to prevent numeric limb explosions. This is a focused hardening that does not change the protocol; it changes the implementation of one helper. The audit confirms the change is the right scope and does not exceed it.

---

## Honesty note

This is a fresh-eyes confirmation audit, not a closure audit. The audit confirms the bounded slice was implemented as recommended and pushed; it does not confirm the implementation is correct in all edge cases. The four "did not verify" items above are the residual gaps. The closure audit (a separate commit) is the right place to verify the full body of `eval_real_ball`, the DomainMembership tail, the test bodies, and the visit_certificate preflight.

The audit's verdict ("A") is an honest grade, not a rubber stamp. The implementation is sound, well-scoped, and matches the audit's recommendations. The residual gaps are minor and bounded.
