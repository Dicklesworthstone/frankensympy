# WS11 Certified Numerics and Algebraic Numbers — Acceptance & Mutation Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws11-certified-numerics-2gu`
**Audit Scope:** Verification of certified numeric enclosures (`RealBall`), algebraic number root isolation and refinement via Sturm sequences (`AlgebraicNumber`), directed-rounding weakening mutant corpus, and gate receipt validation (`gate://ws11-certified-numeric`).

---

## 1. Executive Summary & Acceptance Evidence

All acceptance criteria for `fra-ws11-certified-numerics-2gu` have been implemented and verified:

### Criterion 1: Zero Test Failures Across `fsym-core` and Certified Numerics
```bash
cargo test -p fsym-core
```
**Outcome:** **121 tests passed, 0 failed, 0 ignored**:
- `fsym_core` unit & integration suite: 99 tests passed
- `deterministic_identity`: 1 test passed
- `directed_rounding_mutation`: 16 tests passed
- `fresh_process_id_stability`: 5 tests passed

### Criterion 2: Directed-Rounding and Soundness Mutants Killed
Every certified numerical operation has registered weakening and directed-rounding mutants verified killed in `crates/fsym-core/tests/directed_rounding_mutation.rs`:
1. **Square Root Directed Rounding:**
   - `test_mutant_sqrt_lower_bound_ceiling_killed`: Rounding up the lower bound ($L + \varepsilon$) causes $L^2 > x$, violating containment; mutant killed.
   - `test_mutant_sqrt_upper_bound_floor_killed`: Rounding down the upper bound ($U - \varepsilon$) causes $U^2 < x$, violating containment; mutant killed.
2. **Multiplication Enclosure:**
   - `test_mutant_mul_cross_term_omission_killed`: Omitting second-order cross term $r_1 r_2$ under-reports radius on corner products; mutant killed.
   - `test_mutant_mul_signed_midpoint_without_abs_killed`: Missing absolute value on midpoints in radius computation produces invalid/dangerously small radii; mutant killed.
3. **Addition & Subtraction Error Accumulation:**
   - `test_mutant_add_radius_cancellation_killed`: Assuming errors cancel ($|r_1 - r_2|$) excludes corner sums; mutant killed.
   - `test_mutant_sub_radius_subtraction_killed`: Subtracting radii in subtraction excludes extreme points; mutant killed.
4. **Inversion & Division:**
   - `test_mutant_inv_endpoint_reversal_killed`: Naive inversion without reversing endpoints produces inverted intervals ($L > U$); mutant killed.
   - `test_mutant_inv_containing_zero_admitted_killed`: Inverting intervals containing zero fails closed with `BallError::DivisionByZero`.
5. **Integer Powers:**
   - `test_mutant_pow_even_zero_crossing_monotonicity_killed`: Monotonicity assumption on zero-crossing intervals excludes $0^2 = 0$; mutant killed.
   - `test_mutant_pow_negative_interval_even_power_killed`: Even powers of negative intervals correctly map to positive enclosures.
6. **Algebraic Numbers (Sturm Sequence & Root Isolation):**
   - `test_mutant_algebraic_non_square_free_admitted_killed`: Repeated-root polynomials fail square-free preflight closed with `AlgebraicError::NonSquareFreePolynomial`.
   - `test_mutant_algebraic_interval_with_multiple_roots_admitted_killed`: Multi-root isolating intervals rejected fail-closed via exact Sturm sequence sign variations.
   - `test_mutant_algebraic_interval_with_zero_roots_admitted_killed`: Intervals with zero roots rejected fail-closed.
   - `test_mutant_algebraic_bisection_wrong_bracket_killed`: Bisection choosing the non-bracketed subinterval loses root and is caught by sign invariant.
   - `test_algebraic_refine_to_arbitrary_precision`: Arbitrary precision refinement confirmed down to $\le 10^{-6}$ while maintaining strict root enclosure.
   - `test_algebraic_exact_sign_determination`: Exact sign determination verified for zero, positive roots, and negative roots.

---

## 2. Gate Receipt Verification

The `ws11-certified-numeric` gate was integrated into `xtask` and verified:
```bash
cargo run -p xtask --bin xtask -- gate ws11-certified-numeric
cargo run -p xtask --bin gate-receipt-validator -- artifacts/audit/receipts/ws11-certified-numeric.receipt.json
```
**Receipt:** `artifacts/audit/receipts/ws11-certified-numeric.receipt.json`
- Status: `passed`
- Checks: 4 (no-unsafe scan, `directed_rounding_mutation` test suite, `fsym-core` test suite, `fsym-proof-kernel` test suite)
- Independent Validator Verdict: **ACCEPT**
