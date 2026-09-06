# WS07 Verified Rewriting and Simplification — Acceptance & Mutation Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws07-rewrite-lez`
**Audit Scope:** Verification of deterministic local rewrites, algebraic simplification, kernel-verified derivation trees, and evidence emission in `fsym-simplify`.

---

## 1. Executive Summary & Acceptance Evidence

All acceptance criteria for `fra-ws07-rewrite-lez` have been verified:

### Criterion 1: Zero Failures in `fsym-simplify`
```bash
cargo test -p fsym-simplify
```
**Outcome:** **22 tests passed, 0 failed, 0 ignored**
- Basic simplification & identity reduction (`test_simplify_basic`, `normal_form_is_idempotent`).
- Identity absorption and constant folding (`rewrite_catalog_applies_rules`).
- Trigonometric and hyperbolic zero evaluations (`verified_simplify_uses_the_dedicated_trig_zero_rule`).
- Elementary function evaluations at identity points (`verified_simplify_supports_elementary_functions`).
- Pythagorean identity folds with dedicated proof rule (`verified_simplify_proves_pythagorean_folds_with_dedicated_rule`, `simplify_folds_pythagorean_pairs`, `simplify_preserves_non_pythagorean_adds`).
- Safe points and resource accounting (`budgeted_simplify_stops_atomically_at_safe_points`).
- Bounded expansion and combinatorial growth checks (`expansion_refuses_combinatorial_term_growth_before_allocation`, `expand_binomial_beyond_envelope_is_typed_refusal`).

### Criterion 2: $x \cdot 0$ Evaluates to 0 with Kernel-Verified Derivation; Mutants Killed
Verified in `test_verified_simplify_mul_zero_with_derivation_and_mutants_killed` and `test_simplify_mul_zero`:
1. **Kernel-Verified Derivation:**
   - Evaluates $x \cdot 0 \to 0$, $0 \cdot x \to 0$, $x \cdot y \cdot 0 \to 0$, and $0 \cdot 5 \cdot x \to 0$.
   - Generates a full `DerivationTree` with step-by-step equality claims.
   - Derivations verified independently by `verify_derivation_independent(&derivation, &context)` without generator state.
   - Envelopes verified via `envelope.verify_integrity()`.
2. **Killed Weakening & Forgery Mutants:**
   - **Forged Claim Mutant:** Tampering the derivation step claim to assert $x \cdot 0 = 1$ is caught and rejected by the independent verifier.
   - **Tampered Envelope Mutant:** Tampering the envelope's top-level claim to assert a non-zero RHS fails integrity verification.
   - **Indeterminate Factor Guard:** Expressions multiplying zero by indeterminate or singular terms ($0 \cdot \infty$, $0 \cdot \text{NaN}$, $0 \cdot \tilde{\infty}$, $0 \cdot x^{-1}$) strictly refuse naive zero collapse and preserve the non-zero structure.
