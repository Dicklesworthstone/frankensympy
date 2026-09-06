# WS03 Exact Arithmetic Substrate — Differential Corpus & Safety Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** fra-ws03-exact-arithmetic-7bj
**Audit Scope:** Verification of pure-Rust exact arithmetic (`fsym-bigint`, `fsym-rational`, `fsym-modular`), scalar reference lanes, cancellation safety points, and gate compliance.

---

## 1. Executive Summary & Verification Evidence

All pure-Rust exact arithmetic crates under WS03 were executed under pinned toolchains and tested via remote compilation (`rch`):

```bash
cargo test -p fsym-bigint -p fsym-rational -p fsym-modular
```
**Outcome:** **191 tests passed, 0 failed** across the three crates:
- `fsym-bigint`: 89 passed, 0 failed
- `fsym-rational`: 46 passed, 0 failed
- `fsym-modular`: 56 passed, 0 failed

In addition, the foundation gate in `xtask` was run and verified:
```bash
cargo run -p xtask --bin xtask -- gate foundation
cargo run -p xtask --bin gate-receipt-validator -- artifacts/audit/receipts/foundation.receipt.json
```
**Outcome:** `ACCEPT artifacts/audit/receipts/foundation.receipt.json gate=foundation status=passed checks=7`

---

## 2. Scalar Reference & Differential Lane Verification

Per the non-negotiable constitution rules (Articles VI & XVI) and WS03 deliverables, every optimized arithmetic path is differential-tested against an independent scalar reference lane:

### 2.1 BigInt Multiplication & Division
- **Multiplication Strategies:** `Strategy::SchoolbookReference` vs `Toom3` vs `Karatsuba` vs `NTT-CRT`.
  - In `crates/fsym-bigint/src/ntt.rs`:
    `inverse_round_trip_and_crt_product_match_independent_scalar_lane` verifies that the NTT-CRT product matches `scalar_product` over arbitrary lengths.
  - In `crates/fsym-bigint/src/lib.rs`:
    - `scalar_nth_root_floor` & `caller_degree_roots_match_an_independent_scalar_floor_oracle`: 128-bit scalar reference tests floor nth-root over randomized broad spans.
    - `scalar_greatest_perfect_power` & `greatest_perfect_power_matches_an_independent_bounded_scalar_oracle`: compares against bounded scalar oracle.
    - Ties-to-even and floating-point conversion tests against exact scalar integers (`ties-to-even fixture`).

### 2.2 Rational Arithmetic
- In `crates/fsym-rational/src/lib.rs`:
  - `owned_and_metered_scalar_lanes_match_naive_canonical_arithmetic`: tests metered operations against unmetered naive scalar operations.
  - `governed_constructor_matches_independent_i128_normalization`: verifies coprime canonical reduction against naive 128-bit Euclidean normalization.
  - `rational_order_matches_independent_i128_cross_products`: tests ordering against independent cross-multiplication.
  - `metered_scalar_lanes_match_owned_operators_and_refuse_zero_divisors`: verifies that metered and operator lanes produce identical canonical outputs.

### 2.3 Modular Arithmetic, Reducers, & Primality
- In `crates/fsym-modular/src/lib.rs`:
  - `modular_inverse_matches_bounded_scalar_oracle`: validates extended GCD modular inverse against naive trial inversion.
  - `crt_pair_matches_bounded_exhaustive_oracle`: validates 2-system and multi-system Chinese Remainder Theorem against exhaustive search oracle.
  - `reducers_match_scalar_remainders_over_their_full_admitted_ranges`: Montgomery and Barrett reduction validated against standard scalar division over admitted domains.
  - `finite_field_batch_inverse_matches_scalar_over_generated_batches`: Montgomery batch inverse checked against individual scalar inverses.
  - `prime_stream_and_miller_rabin_match_independent_trial_division`: validates prime stream and deterministic Miller-Rabin against independent scalar trial division.

---

## 3. Resource Governance & Cancellation Safe Points

All WS03 operations respect the `BudgetMeter` resource contract (`fsym-budget`):
- Memory precharging and chunked zero-initialization (capped to 64 KiB chunks) with cancellation checks before each chunk in `ntt.rs`.
- `parse_bytes` preflights input length before UTF-8 decoding to prevent unbounded CPU consumption on malformed inputs.
- `metered_multiply` charges the 64-bit target vector repacks before allocation.
- CRT modulus validation precedes memory cloning or exact arithmetic.
- Cancellation points are tested before and after reservation, between chunks, and upon terminal publication (`new_metered_types_check_cancellation_before_terminal_publication`, `prime_stream_cancellation_is_retry_safe_and_interleavable`).

---

## 4. Workstream Status & Conclusion

WS03 meets all criteria defined in `docs/WORKSTREAM_GRAPH.md` §8 and `registries/workstreams.toml`:
1. Pure Rust, zero unsafe code (`#![forbid(unsafe_code)]` in all 3 crates).
2. Zero C/C++ FFI dependencies.
3. Multiple multiplication strategies differential-tested against scalar reference lanes.
4. Independent foundation receipt verified by `gate-receipt-validator`.
5. 191/191 unit, property, and adversarial tests passing.

Workstream `WS03` is fully validated and ready for closure, unblocking `WS04` ("Terms, domains, assumptions, and bindings").
