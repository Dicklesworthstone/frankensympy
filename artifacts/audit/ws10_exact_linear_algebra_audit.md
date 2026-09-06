# WS10 Exact Linear Algebra Audit

**Bead ID:** `fra-ws10-exact-linear-hbt`  
**Workstream:** WS10 — Exact linear algebra  
**Timestamp:** 2026-09-06  
**Status:** Completed and Verified

---

## 1. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-matrices -> 0 failures`
Executed via RCH remote execution:
```bash
rch exec -- cargo test -p fsym-matrices
```
**Outcome:** Passed. 50/50 unit/integration tests passed, 0 failures, 0 warnings. Strict Clippy check with `-D warnings` on all targets passed cleanly.

### Criterion 2: Eigenvalue claims for deg >= 2 charpolys carry root certificates via Sturm path
Implemented and independently verified:
- `CertifiedEigenvalue`: holds an exact `AlgebraicNumber` root with certified root-isolating ball and algebraic multiplicity.
- `EigenvalueCertificate`: encapsulates the verified monic characteristic polynomial $\det(\lambda I - A)$, the list of certified eigenvalues, total real root count, and total algebraic multiplicity.
- `cauchy_root_bound`: computes a strict upper bound $M \in \mathbb{Q}_{> 0}$ on all root magnitudes of a univariate polynomial.
- `isolate_real_roots_sturm`: bisects over $[-M, M]$ using Sturm sequences, ensuring split points never coincide with roots ($P(m) \ne 0$), isolating each real root into a certified `AlgebraicNumber` ball.
- `Matrix::eigenvalues_with_sturm_certificate`: constructs the characteristic polynomial, verifies it against the matrix at $n + 1$ points, computes square-free decomposition, isolates all real roots across factors, refines any touching/overlapping isolating balls until all balls are strictly disjoint and sorted in ascending order, and constructs the verified certificate.
- `verify_eigenvalue_certificate`: independent reference verifier ensuring:
  1. Monic characteristic polynomial of degree $n$ matches matrix determinant $\det(tI - A)$ at $n + 1$ evaluation points.
  2. Square-free decomposition produces factors $(f_k, m_k)$.
  3. Total real roots and total algebraic multiplicities match independent Sturm count over $(-\infty, \infty)$.
  4. Each eigenvalue defining polynomial divides the characteristic polynomial and matches a square-free factor with equal multiplicity.
  5. Each isolating ball contains exactly 1 real root of its defining polynomial.
  6. All isolating balls are strictly separated and ordered: `balls[i].upper() < balls[i+1].lower()`.
  7. Sum of multiplicities equals total algebraic multiplicity.

---

## 2. Test & Mutant Coverage

Added full test coverage in `crates/fsym-matrices/src/lib.rs`:
- `test_eigenvalues_with_sturm_certificate_2x2_diagonal`: tests $2 \times 2$ diagonal matrix with roots 2 and 5 (multiplicity 1 each) + Serde wire round-trip.
- `test_eigenvalues_with_sturm_certificate_2x2_repeated`: tests $2 \times 2$ repeated eigenvalue 3 with algebraic multiplicity 2.
- `test_eigenvalues_with_sturm_certificate_2x2_rotation_no_real_roots`: tests $2 \times 2$ rotation matrix with $\lambda^2 + 1$ (0 real roots, 0 multiplicity).
- `test_eigenvalues_with_sturm_certificate_3x3_distinct`: tests $3 \times 3$ matrix with distinct eigenvalues 1, 2, 3.
- `test_eigenvalues_with_sturm_certificate_3x3_companion_cubic`: tests $3 \times 3$ companion matrix of $\lambda^3 - 2$ with unique real root $\sqrt[3]{2} \in (1, 2)$.
- `test_eigenvalues_with_sturm_certificate_4x4_repeated`: tests $4 \times 4$ diagonal matrix with repeated roots 1 (mult 2) and 4 (mult 2).
- `test_eigenvalues_sturm_certificate_adversarial_tampering`: kills 10 mutation/forgery categories:
  1. Mutated characteristic polynomial (perturbed constant term) $\to$ rejected.
  2. Inflated eigenvalue multiplicity $\to$ rejected.
  3. Forged total real root count $\to$ rejected.
  4. Forged total algebraic multiplicity $\to$ rejected.
  5. Omitted eigenvalue (dropped real root) $\to$ rejected.
  6. Duplicate eigenvalue $\to$ rejected.
  7. Swapped eigenvalue order (violates ascending order) $\to$ rejected.
  8. Zero multiplicity $\to$ rejected.
  9. Non-square matrix $\to$ rejected with `NotSquare`.
  10. Symbolic entries $\to$ rejected with `UnsupportedCertificateDomain`.

---

## 3. Toolchain & Quality Gates

- `cargo fmt --check`: passed (0 diffs).
- `rch exec -- cargo check -p fsym-matrices`: passed.
- `rch exec -- cargo clippy -p fsym-matrices --all-targets -- -D warnings`: passed.
- `rch exec -- cargo test -p fsym-matrices`: 50/50 passed.
- `./scripts/check.sh registries`: passed (all 19 registries validated).
