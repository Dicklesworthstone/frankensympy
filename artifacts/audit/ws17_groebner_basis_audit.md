# WS17 Groebner Bases and Ideal Algebra Audit

## 1. Workstream Record

- **Workstream ID:** WS17 (`fra-ws17-groebner-jst`)
- **Title:** Groebner bases and ideal algebra
- **Gate:** `gate://ws17-groebner`
- **Receipt:** `artifacts/audit/receipts/ws17-groebner.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-polys -> 0 failures`
- **Result:** `41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings.
- **Formatting:** Clean under `cargo fmt --check`.

### Functional Capabilities Verified:
1. **Buchberger Groebner Basis Algorithm:**
   - Multi-term orders supported: `Lex`, `DegLex`, `DegRevLex`.
   - S-polynomial construction and multi-polynomial division reduction.
   - Minimal and fully reduced basis computation.
   - Resource limits enforced: `MAX_GROEBNER_BASIS_POLYNOMIALS` (1,024) and `MAX_GROEBNER_PENDING_PAIRS` (1,000,000).
2. **Typed Certificate Generation (`GroebnerBasisCertificate`):**
   - Generated via `groebner_basis_with_certificate`.
   - Tracks exact linear combination coefficients (`input_ideal_witnesses`) expressing each Groebner basis polynomial as $\sum_i c_{ij} f_i$.
3. **Independent Reference Verifier (`verify_groebner_certificate`):**
   - **Common Ring:** Validates that all polynomials share common, consistent generator order.
   - **Reverse Containment:** Independently verifies without search that $\sum_i c_{ij} f_i = g_j$.
   - **Forward Containment:** Verifies that every input generator belongs to $\langle G \rangle$ via $f_i \xrightarrow{G} 0$.
   - **Monic Normalization:** Checks that every basis element is monic: $\text{LC}(g_j) = 1$.
   - **Buchberger's S-Polynomial Criterion:** Proves for all pairs $i < j$ that $S(g_i, g_j) \xrightarrow{G} 0$.
   - **Minimal Reduced Support:** Verifies that no monomial in $\text{supp}(g_i)$ is divisible by $\text{LM}(g_j)$ for $i \ne j$.
4. **Ideal Membership & Variable Elimination:**
   - `ideal_membership`: Computes remainder via multivariate division, confirming $f \in I \iff f \xrightarrow{G} 0$.
   - `eliminate`: Computes Lexicographic Groebner basis and filters out polynomials dependent on eliminated variables.

---

## 3. Registered Mutants Killed

Weakening mutants tested and rejected in `test_groebner_basis_certificate_and_verification`:
1. **Non-Monic Basis Element:** Scaling a basis polynomial so $\text{LC} \ne 1 \to$ rejected.
2. **Incomplete Basis:** Omitting a required basis generator $\to$ forward containment / S-pair check fails $\to$ rejected.
3. **Forged Unit Ideal:** Claiming $\langle 1 \rangle$ for a non-trivial ideal $\to$ linear combination witness verification fails $\to$ rejected.
4. **Tampered Witness Identities:** Mutating witness coefficients $\to$ exact identity check fails $\to$ rejected.
5. **Incompatible Generator Ring:** Providing generators from a mismatching ring $\to$ rejected.

---

## 4. Gate Execution and Receipt

- **Gate:** `ws17-groebner`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `tests-fsym-polys`: passed (41/41 tests passing)
