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
- **Unit & Subsystem Tests:** 86 passed; 0 failed.
- **Protocol Gate (`c10_protocol_gate.rs`):** 5 passed; 0 failed.
- **Persistence & Repair Gate (`c9_persistence_repair_gate.rs`):** 6 passed; 0 failed.
- **Cancellation Injection Tests (`cancellation_injection.rs`):** 5 passed; 0 failed.
- **Distribution & Index Gate (`ws16_distribution_index_gate.rs`):** 4 passed; 0 failed.
- **Total Tests:** 106 passed across the crate.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings across all targets.
- **Formatting:** Clean under `cargo fmt --check`.

### Criterion 2: `invalid remote candidate rejected w/o verified-cache pollution (campaign property 11)`
- Verified that adversarial candidate streams containing corrupted derivations, step ID tampering, mismatched claims, and forged result expressions are rejected fail-closed without altering verified cache or workspace state.
- In `test_zero_cache_pollution_under_adversarial_stream`, 10/10 hostile candidates with varied corruptions were rejected with exactly 0 entries admitted to the verified set.

### Functional Capabilities Verified:
1. **Untrusted Remote Worker Execution Lane (`fsym-runtime::remote_worker`):**
   - **Candidate vs Accepted Separation:** Untrusted remote workers produce candidates (`RemoteCandidate`), never accepted state. Worker signatures or votes never substitute for mathematical proof.
   - **Coordinator Verification:** The local `CoordinatorVerifier` independently verifies candidate derivation trees against the immutable assumptions context before accepting.
   - **Privacy-Safe Diagnostics:** Verifier errors expose only categorized diagnostic enums (`RemoteVerificationFailure`), preventing arbitrary untrusted formulas or symbol names from echoing into diagnostics or leaking private context.
   - **Transport Resource Governance:** `MAX_REMOTE_CANDIDATE_BYTES` (1 MB) limits payload allocation prior to parsing; `deny_unknown_fields` fails closed against hostile payload injection.
2. **Semantic Knowledge Graph Indexing (`fsym-runtime::graph_index`):**
   - **Multi-Kind Semantic Nodes:** Indexes `Workspace`, `Symbol`, `Theorem`, and `Derivation` nodes.
   - **Dependency Tracking & Transitive Reachability:** Directed edges with forward and reverse adjacency tracking for exact dependency closure.
   - **Cycle Detection:** Depth-first search with recursion stack tracking detects dependency cycles in polynomial time.
   - **Bit-for-Bit Rebuildability:** Full JSON serialization round-tripping verifies that index deletion and reconstruction preserves graph structure and reachability identically.

---

## 3. Registered Mutants & Boundary Regressions

- **Claim Forgery Refusal:** Attempted result forgery (e.g. valid derivation for reflexivity of $x$, but result claimed as arbitrary integer) is rejected before publication.
- **Task & Context Mismatch:** Remote candidates with mismatched task IDs or differing expected claims are rejected before verifier cycles are spent.
- **Tampered Root Steps:** Derivations with tampered root step IDs fail independent derivation verification.
- **Oversized & Unknown Schemas:** Oversized JSON payloads and unexpected schema fields fail closed at the transport boundary before internal memory allocation.

---

## 4. Gate Execution and Receipt

- **Gate:** `ws16-distribution-index`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `test-ws16-distribution-index-gate`: passed (4/4 tests)
  3. `tests-fsym-runtime`: passed (106/106 tests)
