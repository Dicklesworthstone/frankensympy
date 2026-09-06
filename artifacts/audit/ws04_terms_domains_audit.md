# WS04 Semantic Universe — Deterministic Identity, Domains, & Assumptions Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws04-terms-domains-906`
**Audit Scope:** Verification of the Semantic Term DAG (`fsym-core`), content-addressed deterministic identity (`SEMANTIC-001`), domain separation, 4-valued assumptions (`fsym-assumptions`), typed identifiers (`fsym-id`), and `gate://deterministic-term-identity`.

---

## 1. Executive Summary & Acceptance Evidence

All semantic universe crates under WS04 were verified against the exact acceptance criteria:

### Criterion 1: Zero Test Failures Across Semantic Universe
```bash
cargo test -p fsym-core -p fsym-assumptions -p fsym-id
```
**Outcome:** **158 tests passed, 0 failed, 0 ignored**:
- `fsym-core`: 99 unit & integration tests passed
- `fsym-assumptions`: 44 unit & logic tests passed
- `fsym-id`: 14 unit tests passed, 1 compile-fail integration test passed

### Criterion 2: Fresh-Process ID Stability
```bash
cargo test -p fsym-core --test fresh_process_id_stability
```
**Outcome:** **5/5 tests passed**:
- `fresh_processes_produce_identical_term_ids_and_digests`: Spawns two independent OS child processes (`term_identity_probe`) with distinct address spaces and verifies byte-for-byte identity of all `TermId`s and 32-byte BLAKE3 digests across the canonical suite.
- `de_bruijn_lambdas_alpha_normalize_to_identical_id_and_digest`: Verifies that `Lambda` terms with distinct bound variable names (`\x. x + 1` vs `\y. y + 1`) alpha-normalize to identical `TermId`s and digests.
- `canonical_terms_match_golden_specification`: Validates computed terms against the immutable cross-architecture golden fixture at `artifacts/conformance/fixtures/deterministic_term_identity_v1.json`.
- `insertion_order_does_not_affect_term_identity`: Verifies that forward vs reverse DAG insertion order yields identical composite `TermId`s.
- `mutation_rejection_and_domain_separation`: Verifies that single-bit payload mutations and domain differences change preimages and digests.

### Criterion 3: `gate://deterministic-term-identity` Machine Receipt & Independent Validation
```bash
cargo run -p xtask --bin xtask -- gate deterministic-term-identity
cargo run -p xtask --bin gate-receipt-validator -- artifacts/audit/receipts/deterministic-term-identity.receipt.json
```
**Outcome:** Receipt generated and accepted fail-closed:
```text
ACCEPT artifacts/audit/receipts/deterministic-term-identity.receipt.json gate=deterministic-term-identity status=passed checks=6
```

---

## 2. Deterministic Content Identity Architecture (`SEMANTIC-001`)

Per Constitution Article VII (§7.2, §7.3) and `registries/claims.toml`:

1. **Cryptographic Framing & Domain Tagging:**
   Every term preimage is framed under a constant domain separator `fsym.term.v4\0` followed by the 1-byte `domain.tag()` (`Expression=0`, `Integer=1`, `Rational=2`, `Real=3`, `Complex=4`, `FiniteField=5`). This guarantees that byte preimages in one mathematical domain can never collide with those in another domain.

2. **Truncation & Full Preimage Confirmation:**
   The 64-bit `TermId` is derived from the low 8 little-endian bytes of the 32-byte BLAKE3 digest. A `TermDag` stores both the 64-bit handle and the full 32-byte digest, verifying full preimage equality on interning to eliminate 64-bit birthday collision hazards. Sentinel `0` is strictly reserved and rejected.

3. **Alpha-Equivalence via De Bruijn Lowering:**
   `Lambda` constructs are normalized into de Bruijn indexed bodies (`TermNode::Bound(u32)`). Parameter names are stored as a sidecar for human-readable lifting and display, but are excluded from the hash preimage. Consequently, mathematically identical functions with different dummy parameter names have identical identities.

4. **Cross-Architecture Reproducibility:**
   All integer payloads (including arbitrary-precision `BigInt`) and IDs are serialized into hash preimages in canonical little-endian byte orders with length prefixes. Preimages are independent of host endianness, pointer width, memory addresses, and task scheduling.

---

## 3. Assumptions & Sort Lattice Soundness

1. **Four-Valued Logic:**
   Context evaluation yields `True`, `False`, `Unknown`, or `Contradiction`. Per constitutional rule §7.4, `Unknown` never silently coerces to false.
2. **Immutable Snapshots:**
   Assumptions contexts are copy-on-write immutable data structures. Deduction updates produce fresh context handles rather than mutating shared context.
3. **Contradiction Policy:**
   Contradictory contexts refuse to prove arbitrary propositions (ex falso quodlibet is restricted).

---

## 4. Workstream Status & Conclusion

`fra-ws04-terms-domains-906` has satisfied all acceptance criteria:
- Pure Rust, zero unsafe code (`#![forbid(unsafe_code)]`).
- All 158 workspace semantic tests pass with zero failures.
- Fresh-process stability verified across independent OS processes.
- Golden cross-architecture fixture verified.
- `gate://deterministic-term-identity` receipt validated by independent validator.
