# WS07 Verified Rewriting and Simplification — Acceptance & Mutation Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws07-rewrite-lez`
**Audit Scope:** Verification of deterministic local rewriting, polynomial and trigonometric simplification emitting independent proof derivations (`fsym-simplify`), zero absorption rules, and mutant kills.

---

## 1. Executive Summary & Acceptance Evidence

All acceptance criteria for `fra-ws07-rewrite-lez` have been verified:

### Criterion 1: Zero Test Failures Across `fsym-simplify`
```bash
cargo test -p fsym-simplify
```
**Outcome:** **22 tests passed, 0 failed, 0 ignored**:
- Constant folding, like-term collection, expansion, and power reductions.
- Pythagorean identity folds (`sin^2 + cos^2 = 1`, `sec^2 - tan^2 = 1`, etc.).
- Elementary function evaluations at 0 and 1 with dedicated proof kernel rules.
- Safe point budget accounting and bounded expansion limits.
- Indempotency of normal forms (`normal_form_is_idempotent`).

### Criterion 2: Zero Multiplication with Kernel-Verified Derivation & Mutants Killed
In `test_verified_simplify_mul_zero_with_derivation_and_mutants_killed`:
1. **Kernel-Verified Derivations for Zero Multiplication:**
   - Multiplications involving zero (`x * 0`, `0 * x`, `x * y * 0`, `0 * 5 * x`) evaluate to `0` with complete derivation trees.
   - Independent verification via `verify_derivation_independent` succeeds on the exported derivation trees.
   - Verification receipts and evidence envelopes pass structural integrity checks (`envelope.verify_integrity()`).
2. **Mutants Killed:**
   - **Forged derivation claim mutant:** Mutating derivation tree step claim to `x * 0 = 1` is detected and rejected fail-closed by `verify_derivation_independent`.
   - **Tampered envelope receipt mutant:** Mutating envelope claim to `x * 0 = 1` while receipt digest matches derivation fails `envelope.verify_integrity()`.
   - **Indeterminate factor erasure mutant:** Indeterminate or undefined zero factors (`0 * Infinity`, `0 * NaN`, `0 * ComplexInfinity`, `0 * (1/x)`) are preserved and strictly refused from collapsing to 0.

---

## 2. Invariants & Proof Kernel Integration

1. **Independent Verification Separation:**
   Simplification emissions generate derivation trees verified by `fsym-proof-kernel`'s independent verifier without access to `fsym-simplify` state.
2. **Resource Governance & Safe Points:**
   Budget exhaustion halts rewrite loops cleanly via typed errors (`SimplifyError::BudgetExhausted`).
3. **Combinatorial Growth Guards:**
   Binomial and polynomial expansions preflight total term counts and degree limits before allocation.
