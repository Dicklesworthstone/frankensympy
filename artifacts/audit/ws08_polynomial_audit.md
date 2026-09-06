# WS08 Polynomial Representations and Arithmetic — Acceptance & Invariant Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws08-polynomial-so3`
**Audit Scope:** Verification of polynomial representations (univariate dense $\mathbb{Q}[x]$, multivariate sparse $\mathbb{Q}[x_1, \dots, x_n]$), exact arithmetic with metered cancellation accounting, polynomial identity testing (Schwartz-Zippel PIT), Groebner basis computation and elimination, and representation round-trip invariants.

---

## 1. Executive Summary & Acceptance Evidence

All acceptance criteria for `fra-ws08-polynomial-so3` have been verified:

### Criterion 1: Zero Test Failures Across `fsym-polys`
```bash
cargo test -p fsym-polys
```
**Outcome:** **41 tests passed, 0 failed, 0 ignored**
- Dense univariate arithmetic: Euclidean division, remainder, compose, $L_2$ norm, monic scaling, variable shift.
- Exact calculus & elimination: derivation, integration, Sylvester resultant, discriminant.
- Sparse multivariate ring arithmetic: monomial ordering (lexicographic, graded lex, graded reverse lex), multiplication, metered evaluation.
- Groebner bases & algebraic geometry: Buchberger algorithm, ideal membership certificates, variable elimination.
- Polynomial Identity Testing: Schwartz-Zippel randomized PIT with non-identity witness search.
- Square-free and rational root decomposition.

### Criterion 2: Representation Round-Trip Invariant Checks Green
The representation conversion and preservation invariants are validated in `test_polynomial_representation_round_trip_invariants`:
1. **Univariate Round-Trip Invariance:**
   $$\text{Poly} \longrightarrow \text{Expr} \longrightarrow \text{Poly}$$
   Zero, constant, linear, and higher-degree polynomials (e.g., $3x^4 - 2x^2 + 5$) round-trip between `UnivariatePoly` and `Expr` preserving degree, coefficients, variable symbol, and canonical representation.
2. **Multivariate Round-Trip Invariance:**
   Multivariate polynomials across multi-variable rings (e.g., $\mathbb{Q}[x, y]$ with terms $2x^2y + 3xy^2 - 7$) round-trip between `MultivariatePoly` and `Expr` preserving monomial exponent vectors, coefficients, and ring generators.
3. **Dense vs. Sparse Equivalence:**
   Univariate dense evaluations are confirmed identical to multivariate sparse evaluations under single-generator ring representations.

---

## 2. Invariant & Adversarial Protection

1. **Independent Verification of Algebraic Certificates:**
   - Extended GCD / Bézout identity verifier (`test_extended_gcd_bezout_certificate_verified`, `test_bezout_verifier_requires_canonical_monic_gcd`, `test_bezout_verifier_rejects_zero_gcd_for_nonzero_inputs`).
   - Tampered Bézout certificates and tampered factorizations are rejected fail-closed (`test_mutant_tampered_bezout_rejected`, `test_mutant_tampered_factorization_rejected`).
   - Multivariate divisibility and Groebner membership verifiers fail-closed on tampered certificates (`test_multivariate_divisibility_certificate_rejects_non_monic`, `test_multivariate_divisibility_certificate_rejects_incompatible_rings`).
2. **Negative Corpus Identity Refusal:**
   - Identity checker soundly rejects non-identities and produces explicit counterexample witnesses (`test_negative_corpus_identity_refusal`, `test_schwartz_zippel_detects_non_identity_with_witness`).
3. **Preflight Resource & Limb Bounds:**
   - Sylvester matrix dimension capped at $128 \times 128$; rational scalar coefficients capped at 16,384 limbs to prevent unbounded allocation growth (`resultant_refuses_unbounded_numeric_materialization_and_growth`).
   - Multivariate evaluation is metered through caller-owned resource ledgers, capping extreme exponent amplification (`multivariate_evaluation_refuses_extreme_growth_and_unsupported_exponents`, `metered_multivariate_evaluation_matches_exact_result_and_honors_every_budget`).
