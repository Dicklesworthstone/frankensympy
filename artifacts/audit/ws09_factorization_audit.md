# WS09 GCD, Factorization, and Certificates Audit

## 1. Workstream Record

- **Workstream ID:** WS09 (`fra-ws09-factorization-116`)
- **Title:** GCD, factorization, and certificates
- **Gate:** `gate://ws09-factorization`
- **Receipt:** `artifacts/audit/receipts/ws09-factorization.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-polys -> 0 failures`
- **Result:** `41 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings.
- **Formatting:** Clean under `cargo fmt --check`.

### Criterion 2: Factorization & GCD Certificates with Scope and Independent Verification:
1. **Univariate GCD with Monic Bezout Certificates (`fsym-polys::gcd`):**
   - Extended Euclidean algorithm (`extended_gcd`) produces $(U, V)$ satisfying $U \cdot A + V \cdot B = \gcd(A, B)$.
   - Independent verification (`verify_bezout_certificate`) checks canonical monic normalization, exact linear combination, and divisibility remainders $\text{rem}(A, \gcd) = 0$ and $\text{rem}(B, \gcd) = 0$.
   - Fails-closed: zero GCD is accepted only when both inputs are zero.
2. **Multivariate Common Divisibility Certificates (`fsym-polys::gcd`):**
   - Monic common divisor certificate (`gcd_candidate_with_divisibility_certificate`) independently verified by `verify_multivariate_divisibility_certificate`.
   - Exact divisibility checked: $D \cdot Q_A = A$ and $D \cdot Q_B = B$.
   - Non-monic and cross-ring certificates rejected.
3. **Square-Free Decomposition Certificates (`fsym-polys::factorization`):**
   - Yun's square-free decomposition algorithm produces $P(x) = c \cdot \prod f_i(x)^{e_i}$.
   - Independent verification (`verify_square_free_product_decomposition`) checks:
     - Exact reconstruction: $c \cdot \prod f_i^{e_i} = P(x)$.
     - Monicity and positive-degree factor conventions.
     - Square-freeness: $\gcd(f_i, f_i') = 1$ for all $i$.
     - Pairwise coprimality: $\gcd(f_i, f_j) = 1$ for all $i \neq j$.
4. **Bounded Rational-Root Product Decomposition (`fsym-polys::factorization`):**
   - Rational linear root isolation and quadratic discriminant splitting.
   - Preserves remaining square-free components with clear non-completeness documentation.

---

## 3. Registered Mutants & Boundary Regressions

- **Bezout Certificate Tampering:** Tampered Bezout polynomials, zero GCDs for nonzero inputs, non-monic associates, and incompatible generator symbols are refused.
- **Factorization Tampering:** Mutated factor multiplicities, non-monic factors, zero multiplicities, degree overflow, and non-coprime factor pairs are rejected.
- **Multivariate Ring Mismatch:** Divisibility certificates over mismatched generator lists fail closed.
- **Univariate Wire Boundaries:** Noncanonical wire representations with trailing zeros and exponents exceeding `u32::MAX` fail deserialization.

---

## 4. Gate Execution and Receipt

- **Gate:** `ws09-factorization`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `tests-fsym-polys`: passed (41/41 tests)
