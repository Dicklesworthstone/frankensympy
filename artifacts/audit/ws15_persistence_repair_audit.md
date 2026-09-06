# WS15 Persistence, Checkpoints, and RaptorQ Repair Audit

## 1. Workstream Record

- **Workstream ID:** WS15 (`fra-ws15-persistence-repair-0v8`)
- **Title:** Persistence, checkpoints, and RaptorQ repair
- **Gate:** `gate://ws15-persistence-repair`
- **Receipt:** `artifacts/audit/receipts/ws15-persistence-repair.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-runtime -> 0 failures`
- **fsym-runtime unittests:** `86 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **c9_persistence_repair_gate:** `6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **c10_protocol_gate:** `5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **cancellation_injection:** `5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **Total Tests:** 102 passed across all unit and integration targets.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings across all targets.
- **Formatting:** Clean under `cargo fmt --check`.

### Criterion 2: C9 Crash-Matrix and Repair Trust-Chain
- Executed through `tests/c9_persistence_repair_gate.rs` and validated via `gate://ws15-persistence-repair`.

### Functional Capabilities Verified:
1. **Crash Injection at Publication Steps (`fsym-runtime::checkpoint`):**
   - Truncated writes (10%, 25%, 50%, 75%, 90%, and single-byte truncation) simulate power-loss or crash mid-publication.
   - Truncated bytes fail-closed before state admission; no corrupted or incomplete checkpoint can be resumed.
2. **RaptorQ Repair Envelope (`fsym-runtime::repair`):**
   - Implements RFC 6330 SystematicEncoder and InactivationDecoder.
   - Tested under zero-loss, 1-symbol loss, and multi-symbol loss (4 missing source symbols out of 16).
   - Bit-for-bit payload recovery verified against the canonical BLAKE3 digest.
   - Corrupted source symbols fail digest preflight (`SourceSymbolDigestMismatch`) before decode.
   - Insufficient repair symbols return typed `InsufficientSymbols` error.
3. **Schema and Dependency Integrity (`fsym-runtime::checkpoint`):**
   - Tampered schema version refuses resume.
   - Tampered payload refuses resume due to BLAKE3 digest mismatch.
   - Inflated or altered remaining budget allowances refuse resume.
4. **Append-Only Ephemeral Ledger (`fsym-runtime::ledger`):**
   - Bounded append-only hash chain tracks sequence history and previous record hashes.
   - Direct checkpoint linkage (`append_checkpoint`) verified.
   - Individual record and full chain integrity verification (`verify_chain`) detects any intermediate payload or hash tampering.
5. **Separation of Proof Replay from Raw Byte Repair:**
   - Byte-level RaptorQ recovery does not imply mathematical truth.
   - Repaired derivations must still be independently checked by the reference proof kernel against expected claims.
   - Mismatched or false claims are rejected by `verify_derivation_independent`.
6. **Persistence-Disabled Semantic Equivalence:**
   - Direct symbolic execution without persistence/logging produces the exact same mathematical value as execution with persistence enabled.
   - Database/storage IDs never enter `TermId` or proof identity.

---

## 3. Gate Execution and Receipt

- **Gate:** `ws15-persistence-repair`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `test-c9-persistence-repair-gate`: passed (6/6 tests)
  3. `tests-fsym-runtime`: passed (96/96 tests)
