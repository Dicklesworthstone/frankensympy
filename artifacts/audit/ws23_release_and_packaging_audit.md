# WS23 Packaging, Release, and Certification Gate Audit

**Workstream:** WS23 (`fra-ws23-release-2b6`)  
**Gate:** `gate://ws23-release` / `scripts/check.sh all`  
**Profile:** `sympy-1.14.0-cpython`  
**Date:** 2026-09-06  
**Status:** REFUSED (Fail-Closed by Design, Shrinking Named Blockers Reconciled)  

---

## 1. Executive Summary

This audit establishes the packaging, release readiness, and honest certification boundaries for FrankenSymPy in compliance with `AGENTS.md` §0, §2, §3, §5, §6, Constitution Article XXIV, and `docs/LOCAL_FIRST_RELEASE_AND_VALIDATION.md`.

In accordance with the acceptance criteria for `fra-ws23-release-2b6`:
> `./scripts/check.sh all -> continues to refuse with the shrinking named-blocker list; each closed blocker cites its gate bundle`

Workstream WS23 verifies the following:
1. **Packaging Consistency Enforced:** `scripts/check.sh packaging-consistency` verifies that `scripts/build_python_extension.sh` produces the `fsym_python` cdylib declared in `pyproject.toml`, installs cleanly, and executes symbolic operations (`sympy.diff(x**3, x) -> 3*x**2`) without hidden fallback to upstream SymPy.
2. **Reconciliation of Closed Obligations with Landed Gate Bundles:**
   - `O-SAFETY`: Evidenced by `#![forbid(unsafe_code)]` across all 25 crates, dependency audit (`tools/verify_dependency_and_safety.py`), and pinned toolchain (`rust-toolchain.toml`).
   - `O-EVIDENCE`: Evidenced by `registries/claim_lattice.toml`, `registries/claims.toml`, and validated evidence promotion tiers.
   - `O-VERIFIER`: Evidenced by portable verifier test suite in `fsym-proof-kernel` passing independently of generator crates.
   - `O-ARTIFACT`: Evidenced by `gate://ws15-persistence-repair` receipt (`artifacts/audit/receipts/ws15-persistence-repair.receipt.json`) and audit report.
   - `O-WORKSPACE`: Evidenced by `gate://ws14-agent-protocol` receipt (`artifacts/audit/receipts/ws14-agent-protocol.receipt.json`) and audit report.
   - `O-GRAPH`: Evidenced by `gate://ws16-distribution-index` receipt (`artifacts/audit/receipts/ws16-distribution-index.receipt.json`) and audit report.
   - `O-PYTHON`: Evidenced by `gate://python-object-model` and `gate://ws21-profile-closure` receipts (`artifacts/audit/receipts/ws21-profile-closure.receipt.json`), exclusion ledger, and 230/230 admitted conformance corpus cases.
3. **Blocker Reduction:** Incomplete release blockers shrank from 12 open cross-cutting obligations to 5 remaining open obligations (`O-MONITORS`, `O-PACKAGING`, `O-PERFORMANCE`, `O-PORTFOLIO`, `O-RELEASE`).
4. **Honest Fail-Closed Readiness:** The release candidate orchestrator (`scripts/check.sh all` / `release_readiness`) continues to refuse with exit code 2, honestly reporting that full 1.0 certification, distribution wheels, and machine matrix validation remain open.

---

## 2. Release Blocker Ledger

The active release gate checklist reports the exact remaining blockers:

```
release readiness blockers:
  - cross-cutting obligation 'O-MONITORS' remains a release blocker
  - cross-cutting obligation 'O-PACKAGING' remains a release blocker
  - cross-cutting obligation 'O-PERFORMANCE' remains a release blocker
  - cross-cutting obligation 'O-PORTFOLIO' remains a release blocker
  - cross-cutting obligation 'O-RELEASE' remains a release blocker
  - no immutable compatibility target is certified
  - no resolver-transparent replacement packaging profile is certified
  - quality gate activation 'measurement_schema_frozen' is not satisfied
  - quality gate activation 'monitor_implementation_exists' is not satisfied
  - quality gate activation 'representative_test_inventory_exists' is not satisfied
  - quality gate measurement 'coverage' is not implemented
  - quality gate measurement 'flake' is not implemented
  - quality gate measurement 'runtime' is not implemented
  - quality_gates.toml is not enforced
  - release claim 'RELEASE-001' has no evidence bundle
  - release claim is not certified
  - release gate registry is not implemented
```

---

## 3. Constitutional Integrity Verification

In strict compliance with `AGENTS.md` §2 and §5:
- No claim in `registries/claims.toml` has been inflated to `certified`.
- Claim `RELEASE-001` remains in `planned` status until a complete drop-in release wheel matrix, reproducibility proof, and cryptographic signatures exist.
- No golden files or test suites were regenerated to hide mismatches.
- All 19 machine gate receipts across WS00–WS22 are structurally validated by `gate-receipt-validator` and committed under `artifacts/audit/receipts/`.
