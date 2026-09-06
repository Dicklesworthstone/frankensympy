# WS10 Exact Linear Algebra Audit

## 1. Workstream Record

- **Workstream ID:** WS10 (`fra-ws10-exact-linear-hbt`)
- **Title:** Exact linear algebra
- **Gate:** `gate://ws10-exact-linear`
- **Receipt:** `artifacts/audit/receipts/ws10-exact-linear.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-matrices -> 0 failures`
- **Result:** `50 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings.
- **Formatting:** `cargo fmt --check` clean.

### Criterion 2: Eigenvalue claims for deg>=2 charpolys carry root certificates via Sturm path
- Implemented `Matrix::eigenvalues_with_sturm_certificate`:
  - Computes exact characteristic polynomial $P(\lambda) = \det(\lambda I - A)$.
  - Verifies $P(\lambda)$ via multi-point elimination reference lane (`verify_charpoly_certificate`).
  - Performs square-free decomposition $P(\lambda) = \prod f_k^{m_k}$.
  - Isolates real roots of each square-free factor using Sturm sequence sign variations within Cauchy root bounds (`isolate_real_roots_sturm`).
  - Encloses each real root in an exact [`AlgebraicNumber`] with a certified root isolating [`RealBall`].
  - Refines isolating balls using bisection (`refine_step`) until each ball is tight and pairwise strictly separated.
  - Computes certified algebraic multiplicities and total real root count.
  - Returns a self-contained [`EigenvalueCertificate`].
- Implemented independent reference verifier `verify_eigenvalue_certificate`:
  - Validates square matrix with exact rational coefficients.
  - Verifies characteristic polynomial against matrix.
  - Independently decomposes charpoly into square-free factors.
  - Independently computes Cauchy bounds and counts real roots across all factors via Sturm sequences.
  - Checks that `total_real_roots` and `total_algebraic_multiplicity` strictly match the Sturm reference count.
  - Verifies that each eigenvalue's defining polynomial divides the characteristic polynomial.
  - Verifies that each eigenvalue's isolating ball contains exactly 1 real root of its defining polynomial.
  - Verifies that each eigenvalue's algebraic multiplicity matches the corresponding square-free factor.
  - Verifies that all isolating balls are strictly non-overlapping and ordered: `ball[i].upper() < ball[i+1].lower()`.
- Validated on matrices with:
  - Degree 2 diagonal and repeated eigenvalues (`test_eigenvalues_with_sturm_certificate_2x2_diagonal`, `test_eigenvalues_with_sturm_certificate_2x2_repeated`).
  - Degree 2 rotation matrix with no real roots (`test_eigenvalues_with_sturm_certificate_2x2_rotation_no_real_roots`).
  - Degree 3 distinct eigenvalues (`test_eigenvalues_with_sturm_certificate_3x3_distinct`).
  - Degree 3 companion matrix with irrational cubic root $\sqrt[3]{2}$ (`test_eigenvalues_with_sturm_certificate_3x3_companion_cubic`).
  - Degree 4 diagonal matrix with repeated eigenvalues (`test_eigenvalues_with_sturm_certificate_4x4_repeated`).

---

## 3. Registered Weakening Mutants Killed

Ten distinct adversarial certificate tampering mutations were tested and verified rejected in `test_eigenvalues_sturm_certificate_adversarial_tampering`:
1. **Mutated Charpoly:** Annihilating polynomial altered by adding a constant offset $\to$ rejected.
2. **Forged Real Root Count:** Total real roots inflated $\to$ rejected.
3. **Forged Algebraic Multiplicity:** Total algebraic multiplicity deflated $\to$ rejected.
4. **Forged Root Defining Polynomial:** Defining polynomial replaced with non-dividing polynomial $\to$ rejected.
5. **Root Outside Isolating Ball:** Ball shifted away from the true root $\to$ rejected.
6. **Mismatched Multiplicity:** Eigenvalue multiplicity altered $\to$ rejected.
7. **Overlapping Isolating Balls:** Ball artificially widened to overlap adjacent eigenvalue $\to$ rejected.
8. **Inverted Ball Bounds:** Ball with lower > upper $\to$ rejected.
9. **Truncated Eigenvalue List:** Eigenvalue omitted from certificate $\to$ rejected.
10. **Symbolic Matrix Refusal:** Matrix with symbolic entries passed to exact rational certificate verifier $\to$ rejected with `UnsupportedCertificateDomain`.

Additional certificate verification families tested and passing in `fsym-matrices`:
- LU certificate verifier: permutation orthogonality, unit lower-triangular, upper-triangular, $PA = LU$.
- QR certificate verifier: $Q^T Q$ diagonal orthogonality, $R$ unit upper-triangular, $QR = A$.
- LDL certificate verifier: $L$ unit lower-triangular, $D$ diagonal, $LDL^T = A$.
- Matrix Inverse certificate verifier: $A A^{-1} = I$ and $A^{-1} A = I$.
- Nullspace certificate verifier: $A v = 0$ for all basis vectors, linear independence of basis, rank-nullity theorem.

---

## 4. Gate Execution and Receipt

The gate was executed via `cargo run -p xtask --bin xtask -- gate ws10-exact-linear` and verified with `gate-receipt-validator`:
- **Gate:** `ws10-exact-linear`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `tests-fsym-matrices`: passed (50/50 tests passing)
