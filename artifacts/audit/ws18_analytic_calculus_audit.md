# WS18 Integration, Limits, Series, and Transforms Audit

## 1. Workstream Record

- **Workstream ID:** WS18 (`fra-ws18-analytic-calculus-vop`)
- **Title:** Integration, limits, series, and transforms
- **Gate:** `gate://ws18-analytic-calculus`
- **Receipt:** `artifacts/audit/receipts/ws18-analytic-calculus.receipt.json`
- **Status:** Closed / Complete

---

## 2. Acceptance Criteria Verification

### Criterion 1: `cargo test -p fsym-calculus -> 0 failures`
- **Result:** `63 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out` (across library and integration suites).
- **Clippy:** Strict workspace-wide `-D warnings` passed with 0 warnings.
- **Formatting:** Clean under `cargo fmt --check`.

### Functional Capabilities Verified:
1. **Symbolic Integration:**
   - Elementary rule-based antiderivative construction: power rule, exponentials, trigonometrics, hyperbolics, and logarithm.
   - Bounded tabular integration by parts for products of polynomials (degree $\le 8$) with analytic functions of linear arguments ($\exp, \sin, \cos, \sinh, \cosh$).
   - Definite integration with strict endpoint and singularity admission checks.
   - Conservative typed refusals (`IntegrationFailed`, singularity detection) ensuring no silent incorrect evaluations.
2. **Limits & Asymptotics:**
   - Finite limits with pole detection propagating through nested function applications.
   - Infinite limit evaluation via degree analysis of polynomial and rational expressions.
   - Bounded substitution traversal limits (16,384 nodes, depth 128) preventing runaway recursion.
   - Refusal of unresolvable indeterminate forms (e.g., $0/0$) returning typed `Undetermined`.
3. **Taylor & Power Series:**
   - Order-capped Taylor polynomial series expansion.
   - Coefficient pole validation rejecting singular expansions at singular expansion points.
   - High-precision numerical agreement near expansion origins.
4. **Integral Transforms (Laplace & Fourier):**
   - Elementary Laplace transform catalog ($t^n$, $\exp(at)$, hyperbolic, damped trig).
   - Exact numeric Region of Convergence (ROC) tracking and composition.
   - Fourier transform decay rate admission ensuring genuine $L^1$ or decaying rate evidence before transformation.

---

## 3. Registered Mutants & Boundary Regressions

- **Singular Definite Integrals:** Integrands with poles on or within integration intervals (e.g. $x^{-2}$ over $[-1, 1]$) are refused rather than incorrectly evaluating to negative values.
- **Nested Pole Limits:** $\exp(1/x)$ at $x \to 0$ correctly detects the interior pole and refuses direct substitution.
- **Singular Taylor Series:** $x^{-1}$ and $\exp(1/x)$ at $x_0 = 0$ fail closed instead of producing corrupt series polynomials.
- **Transform ROC Soundness:** Symbolic/unproved decay rates and multi-term order-dependent sums are refused without rigorous rate bounds.
- **Special Functions Boundary:** Binomial coefficients with negative integer lower indices evaluate to exact zero before equal-argument shortcuts.

---

## 4. Gate Execution and Receipt

- **Gate:** `ws18-analytic-calculus`
- **Profile:** `sympy-1.14.0-cpython`
- **Status:** `passed`
- **Checks:**
  1. `workspace-no-unsafe`: passed
  2. `tests-fsym-calculus`: passed (63/63 tests passing)
