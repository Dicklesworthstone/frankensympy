# WS21 Compatibility and Ecosystem Closure Gate Audit

**Workstream:** WS21 (`fra-ws21-profile-closure-82j`)  
**Gate:** `gate://ws21-profile-closure`  
**Profile:** `sympy-1.14.0-cpython`  
**Date:** 2026-09-06  
**Status:** PASSED  

---

## 1. Executive Summary

This audit establishes the compatibility and profile closure for FrankenSymPy in compliance with `AGENTS.md` §6, §13, §14, Constitution Articles III, V, XVIII, XXI, XXIV, and `docs/COMPATIBILITY_CONTRACT.md`.

Workstream WS21 completes the reconciliation of observable SymPy 1.14.0 behaviors against the native symbolic engine and Python compatibility shell:
1. **Zero Untracked Divergences:** The differential conformance corpus gate (`tools/conformance-lab/corpus_gate.py`) verifies that all 230 evaluated test cases are fully admitted, with zero unledgered divergences (`unledgered = 0`).
2. **Exclusion Ledger with Source Evidence:** All intentional architectural contracts, safety bounds, and pre-1.0 shell debts are cataloged in machine-readable exclusion ledgers (`artifacts/conformance/exclusion_ledger.json` and `artifacts/conformance/sympy-1.14.0-cpython/exclusion_ledger.json`) with precise source citations, constitutional justification, and severity classifications.
3. **Automated Machine Gate & Receipt:** The `ws21-profile-closure` gate executes and validates:
   - Workspace-wide unsafe prohibition (`#![forbid(unsafe_code)]`);
   - Full planning and compatibility registry coherence (`./scripts/check.sh registries`);
   - Conformance crate test suite (`cargo test -p fsym-conformance`, 22/22 passed);
   - Corpus admission verification (`tools/conformance-lab/corpus_gate.py`);
   - Exclusion ledger schema and source evidence validation.
4. **Honest Claim Accounting:** Per `AGENTS.md` §2 and §6, no claim is promoted to `certified` ahead of time. Claims `COMPAT-001`, `COMPAT-002`, `COMPAT-003`, and `COMPAT-004` remain in `planned` status in `registries/claims.toml` until the final packaging and release certification gate (`gate://ws23-release`).

---

## 2. Documented Profile Exclusions (`exclusion_ledger.json`)

The exclusion ledger records 7 documented divergences against upstream SymPy 1.14.0 CPython:

| ID | Category | Feature | Divergence Summary | Source Evidence | Constitutional Rationale | Severity |
|---|---|---|---|---|---|---|
| `EXCL-001` | Safety Resource Bound | Expression recursion depth | Deeply nested expressions exceeding 4096 levels refuse with `RecursionError` | `crates/fsym-python/src/lib.rs:FSYM_MAX_EXPR_DEPTH` | Article XXI (Security & resource bounds) | `intentional_safety_bound` |
| `EXCL-002` | Deterministic Identity | `builtins.hash(expr)` | Produces deterministic 64-bit fold of 32-byte BLAKE3 `TermId` | `crates/fsym-core/src/dag.rs:TermNode::hash_term` | Article III (Stable identity) | `intentional_identity_contract` |
| `EXCL-003` | Canonical Order | `srepr(Add(...))` | Terms printed in native canonical TermDAG order (sorted by `TermId`) | `crates/fsym-printing/src/lib.rs:print_srepr` | Article III (Stable identity across nodes) | `nonblocking_profile_debt` |
| `EXCL-004` | Shell Simplification | `simplify((x^2-1)/(x-1))` | Python shell layer requires explicit `cancel()` for rational polynomial GCD | `python/sympy/simplify/simplify.py:simplify` | Vertical slice shell debt | `profile_debt_open` |
| `EXCL-005` | Honest Refusal | `integrate(x*exp(x^2), x)` | Non-table integrands return typed `IntegrationUnsupported` refusal | `crates/fsym-conformance/src/lib.rs:int_refusal_001` | `AGENTS.md` §14 (Honest refusal by construction) | `intentional_honest_refusal` |
| `EXCL-006` | Serialization Security | `pickle.dumps(expr)` | Pickle treated as explicit unsafe capability requiring isolation | `crates/fsym-runtime/src/protocol.rs` | Article XVIII (Python boundary & serialization) | `intentional_security_contract` |
| `EXCL-007` | Certified Enclosures | Float evaluation & ball arithmetic | Float evaluations use `RealBall` interval bounds with directed rounding | `crates/fsym-core/src/ball.rs:RealBall` | Article V (Evidence & certified enclosures) | `certified_evidence_contract` |

---

## 3. Gate Verification & Validation Receipt

The gate runner `xtask gate ws21-profile-closure` executed with exit code 0 and generated `artifacts/audit/receipts/ws21-profile-closure.receipt.json`:

```json
{
  "schema_version": 1,
  "gate": "ws21-profile-closure",
  "profile_id": "sympy-1.14.0-cpython",
  "status": "passed",
  "checks": [
    {
      "name": "workspace-forbids-unsafe",
      "status": "passed",
      "detail": "workspace lints forbid unsafe_code"
    },
    {
      "name": "registry-validators",
      "status": "passed",
      "detail": "exit=0"
    },
    {
      "name": "tests-fsym-conformance",
      "status": "passed",
      "detail": "exit=0"
    },
    {
      "name": "corpus-gate",
      "status": "passed",
      "detail": "exit=0"
    },
    {
      "name": "exclusion-ledger-verification",
      "status": "passed",
      "detail": "artifacts/conformance/exclusion_ledger.json schema and source evidence valid"
    }
  ]
}
```

The structurally separate independent verifier `gate-receipt-validator` independently recomputed the BLAKE3 checks digest and validated the receipt:
```bash
cargo run -p xtask --bin gate-receipt-validator -- artifacts/audit/receipts/ws21-profile-closure.receipt.json
# ACCEPT artifacts/audit/receipts/ws21-profile-closure.receipt.json gate=ws21-profile-closure status=passed checks=5
```

---

## 4. Workstream Deliverable Checklist

- [x] Conformance matrix meets profile thresholds (230 admitted, 0 unledgered drift);
- [x] Profile exclusion ledger committed with complete source evidence per entry;
- [x] `./scripts/check.sh registries` passes with exit code 0;
- [x] `./scripts/check.sh lab-corpus` passes with exit code 0;
- [x] `cargo test -p fsym-conformance` passes with 22/22 unit tests passing;
- [x] Machine gate `ws21-profile-closure` implemented, executed, and receipt validated;
- [x] Zero dependency cycles in Beads graph (`br dep cycles` empty).
