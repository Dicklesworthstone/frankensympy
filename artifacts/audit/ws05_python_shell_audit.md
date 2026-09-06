# WS05 Python Compatibility Shell Vertical Slice — Differential & Surface Audit

**Author:** PearlTower (Antigravity / Gemini 3.8 Flash)
**Date:** 2026-09-06
**Bead:** `fra-ws05-python-shell-sl7`
**Audit Scope:** Verification of the SymPy-compatible Python shell (`python/sympy`), PyO3 bridge (`crates/fsym-python`), surface test suite (`python/tests`), and differential conformance lab (`tools/conformance-lab`).

---

## 1. Executive Summary & Acceptance Evidence

All Python shell acceptance criteria under WS05 were verified:

### Criterion 1: Surface Test Suite Discovery & Execution
```bash
PYTHONPATH=python .venv-conformance/bin/python -m unittest discover -s python/tests -p 'test_*.py'
```
**Outcome:** **40 tests ran, 40 passed, 0 failures, 0 errors** (Ran in 29.69s).
- Covers Basic, Atom, Expr, Symbol, Dummy, Integer, Rational, Float, Add, Mul, Pow, Function, Derivative, AppliedUndef.
- Covers args, func, free_symbols, as_ordered_terms, as_coeff_Mul, as_coeff_Add, as_base_exp, as_numer_denom.
- Covers Singleton registry `S.Zero`, `S.One`, `S.NegativeOne`, `S.Half`, `S.Pi`, `S.Infinity`, `S.ComplexInfinity`.
- Covers `diff`, `simplify`, `integrate`, `solve`, `dsolve` (honest refusal), `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`.
- Covers pickle protocol-4 round-trip, deepcopy identity preservation for held forms (`evaluate=False`).
- Covers custom Python subclass collapse prevention and safe extension boundary.

### Criterion 2: Conformance Lab Differential Mode vs Live Oracle Goldens
```bash
.venv-conformance/bin/python tools/conformance-lab/capture.py diff tools/conformance-lab/profiles/sympy-1.14.0-cpython-r2.toml --candidate-python .venv-conformance/bin/python3
```
**Outcome:** **14/14 admitted fixtures, 14 type-matched, 0 discrepancies, exit 0**.

### Criterion 3: Shell Infidelity Closure & Corpus Gate
```bash
./scripts/check.sh lab-corpus
```
**Outcome:** **230 admitted, 0 drift, 0 unledgered, 0 open ledger records**.
- All 44 historical discrepancy ledger records closed and verified (`dc5d637`).
- Zero stub module files or faked `__module__` paths.
- Integer(0) constructor interning aligned with SymPy singleton semantics.
- Pow representation and Float cross-type structural equality matched to SymPy 1.14.0 behavior.

### Criterion 4: Machine Gate Receipt & Independent Validation
```bash
cargo run -p xtask --bin xtask -- gate python-object-model --profile sympy-1.14.0-cpython-r2
cargo run -p xtask --bin gate-receipt-validator -- artifacts/audit/receipts/python-object-model.receipt.json
```
**Outcome:** Validated fail-closed:
```text
ACCEPT artifacts/audit/receipts/python-object-model.receipt.json gate=python-object-model status=passed checks=4
```

---

## 2. Architecture & Invariants Verified

1. **Dual-Lane Object Model:**
   Native kernel owns deterministic term semantics; Python wrapper layer maintains Python class identity, `args`/`func` reconstruction, and custom subclass hooks.
2. **Held Forms Preservation:**
   `evaluate=False` constructs expressions without canonical simplification (e.g. `Add(x, x, evaluate=False)` preserves 2 args), correctly surviving deepcopy and pickle round-tripping.
3. **Fail-Closed Execution:**
   Unimplemented functionality (e.g. general ODE `dsolve`) raises explicit `NotImplementedError` rather than returning unverified strings or fallback approximations.
4. **Oracle Isolation:**
   Candidate subprocess runs strictly in isolated virtual environment without oracle package interference, enforced by the `oracle-isolation-probe` gate check.
