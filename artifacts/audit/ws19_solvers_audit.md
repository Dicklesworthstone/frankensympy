# WS19 Solvers, Sets, Logic, and ODE/PDE Audit

## 1. Workstream Record

- **Workstream ID:** WS19 (`fra-ws19-solvers-612`)
- **Title:** Solvers, sets, logic, ODE, and PDE
- **Gate:** `gate://ws19-solvers`
- **Receipt:** `artifacts/audit/receipts/ws19-solvers.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-solvers -p fsym-sets -p fsym-logic -> 0 failures`
- **fsym-solvers:** `24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **fsym-sets:** `21 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **fsym-logic:** `17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`.
- **Total Tests:** 62 passed across all three crates.
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings across all targets.
- **Formatting:** Clean under `cargo fmt --check`.

### Functional Capabilities Verified:
1. **Algebraic & Polynomial Solvers (`fsym-solvers`):**
   - Linear equation solving with rational and symbolic constants.
   - Exact quadratic solving with rational-root specialization and symbolic radical handling.
   - Univariate polynomial solving via bounded rational root decomposition and recursive factorization dispatch.
   - 2-variable polynomial systems solved with Sylvester resultant elimination, non-linear back-substitution checks, and independent polynomial residual verification (`verify_poly_system_solution`).
   - Conservative refusals on non-linear expressions, underdetermined systems, and unsupported degrees.
2. **Ordinary Differential Equations (`fsym-solvers::ode`):**
   - First-order linear ODEs solved via integrating factors with independent residual verifier (`verify_first_order_linear_solution` / `verify_linear_first_order_solution`).
   - Second-order constant-coefficient linear homogeneous and nonhomogeneous ODEs with independent residual verification (`verify_const_coeff_second_order_solution`, `verify_const_coeff_second_order_nonhomogeneous_solution`).
   - Refusal of unverified non-square characteristic radicals until algebraic-number verifiers exist.
   - Cauchy-Euler equations with independent residual verification (`verify_cauchy_euler_solution`).
   - Trust-boundary preflight caps (`verifier_inputs_within_bounds`) bounding expression depth, node counts, fanout, and numeric limb counts to prevent algorithmic resource exhaustion.
3. **Set Theory & Real Topology (`fsym-sets`):**
   - 3-valued set membership logic (True, False, Unknown) preventing false claims on undecidable symbolic bounds.
   - Finite sets, intervals (open, closed, half-open, unbounded), unions, intersections, complements, and differences.
   - Set measure computation requiring proven extended-real ordering.
   - Topological interior, closure, boundary, and open/closed/compact predicates.
   - Structural refusal of non-real or unresolved points, invalid interval interiors, and indeterminate infinity expressions.
   - Involutive double complements and De Morgan complement duality.
4. **Propositional Logic & SAT (`fsym-logic`):**
   - DPLL SAT solver with strict search budget tracking; resource exhaustion returns typed refusal rather than falsifying as UNSAT.
   - Tseitin equisatisfiable CNF transformation avoiding distributive explosion.
   - Model extraction and verification against original boolean formulas.
   - Tautology (`is_valid`) and contradiction detection reusing the existing root variable rather than inflating SAT depth.
   - Truth table generation with rigorous size and depth preflight caps.

---

## 3. Registered Mutants & Boundary Regressions

- **Residual Verifier Boundaries:** Residual verifiers reject malformed polynomial representations, non-canonical zero coefficients, non-linear back substitutions, and oversized expression graphs.
- **ODE Residual Verification:** Tampered ODE solutions (e.g. wrong exponents, wrong constants, inverted signs) are rejected by independent residual verifiers.
- **DPLL Resource Refusal != UNSAT:** Tests explicitly verify that exhausted search budgets return a budget error rather than claiming unsatisfiability.
- **Real Topology Refusals:** Non-real elements (e.g. `{I}`), unresolved symbolic points, and indeterminate infinities are refused rather than silently misclassified.
- **Empty Aggregate Identities:** Serde-constructed empty FiniteSet, Union, and Intersection variants strictly obey empty/universal set identities.

---

## 4. Gate Execution and Receipt

- **Gate:** `ws19-solvers`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `tests-fsym-solvers`: passed (24/24 tests)
  3. `tests-fsym-sets`: passed (21/21 tests)
  4. `tests-fsym-logic`: passed (17/17 tests)
