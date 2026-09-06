# WS14 Agent Protocol and Semantic Workspaces Audit

## 1. Workstream Record

- **Workstream ID:** WS14 (`fra-ws14-agent-protocol-z1s`)
- **Title:** Agent protocol and semantic workspaces
- **Gate:** `gate://ws14-agent-protocol`
- **Receipt:** `artifacts/audit/receipts/ws14-agent-protocol.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-runtime -> 0 failures`
- **fsym-runtime unittests:** `86 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **c10_protocol_gate:** `5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **cancellation_injection:** `5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **Total Tests:** 96 passed across all unit and integration targets.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings across all targets.
- **Formatting:** Clean under `cargo fmt --check`.

### Criterion 2: C10 Protocol Conformance
- Executed through `tests/c10_protocol_gate.rs` and validated via `gate://ws14-agent-protocol`.

### Functional Capabilities Verified:
1. **NDJSON Protocol & Fail-Closed Wire Bounds (`fsym-runtime::protocol`):**
   - Valid command dispatch: `Bind`, `Eval`, and `Diff` operations.
   - Resource preflight: Unknown payload fields fail closed (`malformed_request`) before workspace state mutation.
   - Oversized envelopes exceeding `MAX_AGENT_NDJSON_REQUEST_BYTES` are rejected immediately before JSON parsing or memory allocation.
2. **Workspace Fork, Patch, and Proof-Aware Merge (`fsym-runtime::workspace`):**
   - Semantic workspaces support branch forking and atomic patch application (`WorkspacePatch`).
   - Content digest tracking: merge receipts compute BLAKE3 digests over deterministic sorted bindings, assumptions, and derivations.
   - Proof-aware merge: imported derivations from branch workspaces are verified by the proof kernel before admission to the base workspace.
3. **Negative Corpus Rejection:**
   - **Conflicting Bindings:** Divergent bindings for identical symbols between base and branch fail with `WorkspaceError::BindingConflict`.
   - **Mismatched Assumptions:** Branches with incompatible assumption snapshots fail merge with `WorkspaceError::AssumptionContextMismatch`.
   - **Tampered / Unverified Proofs:** Derivations with tampered step roots fail merge with `WorkspaceError::DerivationVerificationFailed`.
4. **Candidate vs Accepted Separation (`fsym-runtime::remote_worker`):**
   - `CoordinatorVerifier` enforces strict separation: untrusted remote candidates are treated as unverified candidates until checked against task claims.
   - Mismatched task IDs fail immediately with `RemoteWorkerError::TaskMismatch`.
   - Forged results / non-matching claims fail with `RemoteWorkerError::ClaimForgery`.
   - Tampered derivations fail verification with `RemoteWorkerError::VerificationFailed`.
5. **Transcript-Free Deterministic Replay (`fsym-runtime::replay`):**
   - Replay logs record seeds, operation names, and resource-dimension charges.
   - Deterministic replay produces bit-for-bit identical BLAKE3 digest verification without relying on string transcripts.
   - Tampered event payloads or out-of-order event indices fail `verify_integrity()`.

---

## 3. Gate Execution and Receipt

- **Gate:** `ws14-agent-protocol`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `test-c10-protocol-gate`: passed (5/5 tests)
  3. `tests-fsym-runtime`: passed (91/91 tests)
