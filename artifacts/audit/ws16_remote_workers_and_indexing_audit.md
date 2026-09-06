# WS16 Remote Workers and Graph Indexing Audit

## 1. Workstream Record

- **Workstream ID:** WS16 (`fra-ws16-distribution-index-grc`)
- **Title:** Remote workers and graph indexing
- **Gate:** `gate://ws16-distribution-index`
- **Receipt:** `artifacts/audit/receipts/ws16-distribution-index.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-runtime -> 0 failures`
- **Unit & Module Tests:** 86 tests passed in `fsym-runtime` library.
- **C10 Protocol Gate Suite (`tests/c10_protocol_gate.rs`):** 5 integration tests passed.
- **Cancellation Injection Suite (`tests/cancellation_injection.rs`):** 5 integration tests passed.
- **WS16 Distribution & Graph Index Suite (`tests/ws16_distribution_index_gate.rs`):** 4 integration tests passed.
- **Total:** 100 passed; 0 failed; 0 ignored; 0 measured.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings.
- **Formatting:** Clean under `cargo fmt --check`.

### Criterion 2: Invalid remote candidate rejected w/o verified-cache pollution (Campaign Property 11)
- **Gate:** `ws16-distribution-index` executed via `xtask gate ws16-distribution-index`.
- **Adversarial Stream Verification (`test_zero_cache_pollution_under_adversarial_stream`):**
  - 10 hostile remote candidates with corrupted proof derivation trees were submitted to `CoordinatorVerifier`.
  - Zero unverified entries were admitted to accepted state (accepted: 0, rejected: 10).
- **Receipt:** `artifacts/audit/receipts/ws16-distribution-index.receipt.json` validated fail-closed with `gate-receipt-validator` (3/3 checks passed).

---

## 3. Functional Capabilities & Verification Highlights

1. **Untrusted Remote Candidate Lane (`fsym-runtime::remote_worker`):**
   - Remote worker results remain untrusted candidates until local independent verification by `CoordinatorVerifier`.
   - Task ID mismatches are rejected immediately before spending verifier work (`RemoteWorkerError::TaskMismatch`).
   - Claim forgery (where claimed result does not match the verified claim root) is detected and rejected (`RemoteWorkerError::ClaimForgery`).
   - Tampered derivation proofs fail proof kernel verification (`RemoteWorkerError::VerificationFailed`).

2. **Wire Decoding Bounds & Schema Strictness:**
   - Enforces `deny_unknown_fields` during candidate deserialization (`RemoteCandidateWire`), failing closed on unknown fields.
   - Rejects payloads exceeding `MAX_REMOTE_CANDIDATE_BYTES` (1MB) before parsing (`RemoteWorkerError::PayloadTooLarge`).

3. **Semantic Knowledge Graph Indexing (`fsym-runtime::graph_index`):**
   - Indexes entities across multiple kinds (`Workspace`, `Symbol`, `Theorem`, `Derivation`).
   - Transitive dependency reachability via breadth-first search.
   - Cycle detection via depth-first search recursion stack tracking.
   - Serializes and rebuilds bit-for-bit, preserving identical graph structure and reachability.

---

## 4. Gate Execution and Receipt Summary

- **Gate:** `ws16-distribution-index`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-forbids-unsafe`: passed
  2. `test-ws16-distribution-index-gate`: passed
  3. `tests-fsym-runtime`: passed
