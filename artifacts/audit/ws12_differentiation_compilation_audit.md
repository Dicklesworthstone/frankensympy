# WS12 Differentiation and Symbolic Compilation Audit

## 1. Workstream Record

- **Workstream ID:** WS12 (`fra-ws12-diff-compilation-y05`)
- **Title:** Differentiation and symbolic compilation
- **Gate:** `gate://ws12-certified-jacobian`
- **Receipt:** `artifacts/audit/receipts/ws12-certified-jacobian.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-calculus -> 0 failures`
- **Unit Tests:** 58 unit and integration tests passing in `fsym-calculus`.
- **C7 Hero Suite:** 2 integration tests passing in `tests/sparse_jacobian_c7.rs`.
- **Sparse Gate Suite:** 3 integration tests passing in `tests/sparse_jacobian_gate.rs`.
- **Total:** 63 passed; 0 failed; 0 ignored; 0 warnings.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings.
- **Formatting:** Clean under `cargo fmt --check`.

### Criterion 2: Sparse Jacobian fixture w/ proof replay + compiled evaluator agreement (C7 gate)
- **Sparse Jacobian Implementation (`crates/fsym-calculus/src/sparse_jacobian.rs`):**
  - Analyzes structural symbol dependencies to construct exact `SparsityPattern` coordinates.
  - Performs structural differentiation with verified proof trees (`verified_diff`), attaching a `DerivationTree` to every non-zero entry `SparseJacobianEntry`.
  - Implements distance-1 column coloring to determine minimum color compression factor $k \le n$ such that columns sharing a row non-zero receive distinct colors.
  - Compiles entries into validated bytecode evaluators (`CompiledExpr`) for fast numeric evaluation.
  - Provides sparse, dense, and compressed evaluation methods (`try_eval_sparse`, `try_eval_dense`, `try_eval_compressed`).
- **Independent Proof Replay & Evaluator Agreement Verifier (`verify_sparse_jacobian_certificate`):**
  - Replays and verifies every derivation proof using `verify_diff_derivation` against the proof kernel.
  - Audits omitted coordinates to confirm that any entry not in the sparsity pattern is mathematically zero.
  - Verifies distance-1 graph coloring validity, ensuring no two non-zeros in any row share the same column color.
  - Evaluates both sparse and dense compiled bytecode at multiple test points and confirms bit-for-bit/floating-point agreement within $10^{-12}$.
- **Fixtures Verified:**
  - **4D Hero Sparse Residual Fixture (`tests/sparse_jacobian_c7.rs`):**
    - $f_0(x_0, x_1, x_2, x_3) = x_0^2 + \sin(x_1) - 1$
    - $f_1(x_0, x_1, x_2, x_3) = x_1 x_2 - \cos(x_2)$
    - $f_2(x_0, x_1, x_2, x_3) = \exp(x_2) + x_3^3 - 4$
    - $f_3(x_0, x_1, x_2, x_3) = x_0 x_3 - 2$
    - Sparsity: 8 non-zeros out of 16 (50% density); ring structure compressed to 2 colors.
    - Verified derivation proof replay across all entries.
    - Agreement verified against dense evaluators and Serde round-trip wire decoding.
  - **3D & 4D Tridiagonal Sparse Fixtures (`tests/sparse_jacobian_gate.rs`):**
    - Analytical derivative agreement across multiple evaluation points.
    - Exact 0.0 evaluation verified for structural zero entries.
    - Central finite differences diagnostic consistency confirmed ($h=10^{-6}$, $\text{tol}=10^{-5}$).

---

## 3. Registered Mutants Killed

Adversarial mutations verified rejected in `test_sparse_jacobian_c7_adversarial_tampering` and `test_sparse_jacobian_adversarial_tampering_and_mutants_killed`:
1. **Mutated Proof Claim:** Altered claim in derivation step $\to$ rejected with `ClaimDiscrepancy`.
2. **Mutated Symbolic Derivative:** Forged symbolic derivative expression $\to$ rejected with `ProofFailed`.
3. **Mutated Proof Rule:** Swapped definitional reduction rule $\to$ rejected with `RuleMismatch`.
4. **Omitted Nonzero Entry:** Artificially omitted structural non-zero from sparsity pattern $\to$ caught by omission audit with `OmittedNonzero`.
5. **Coloring Collision:** Forcing conflicting columns to share the same color $\to$ caught with `ColoringCollision`.
6. **Discrepant Bytecode:** Forged constant opcode replacing true derivative bytecode $\to$ caught with `EvaluatorDisagreement`.
7. **Invalid Root Step ID:** Tampering with tree root step ID $\to$ caught with `UnknownStep`.
8. **Buffer Length Mismatch:** Passing undersized residual or Jacobian buffers $\to$ rejected with `ResidualBufferMismatch` / `JacobianBufferMismatch`.
9. **Variable Index Out of Bounds:** Input vector shorter than variable count $\to$ rejected with `VariableOutOfBounds`.

---

## 4. Gate Execution and Receipt

- **Gate:** `ws12-certified-jacobian`
- **Receipt:** `artifacts/audit/receipts/ws12-certified-jacobian.receipt.json`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `test-sparse-jacobian-c7`: passed
  3. `test-sparse-jacobian-gate`: passed
  4. `tests-fsym-calculus`: passed
