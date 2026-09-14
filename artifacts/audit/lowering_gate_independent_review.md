# Independent lowering-gate review (`fra-rc-lowering-gate-ba5`)

## 1. Gate record

- **Gate bead:** `fra-rc-lowering-gate-ba5`, reviews `fra-rc-lowering-8w3` (typed lowering/lifting contract)
- **Reviewer battery:** `crates/fsym-core/tests/symbol_identity_review.rs` + `python/tests/test_lowering_contract_review.py`, authored by BoldGorge (commit `f7966e6`), not the implementation author
- **Gate owner:** CrimsonTurtle (independent of implementer PeachMoose and battery author BoldGorge), 2026-09-14
- **Verdict:** **PASS** on the slice's declared deliverable — every must-hold identity/lifting/receipt obligation is green; 5 defects are **recorded findings** with owners assigned (4 pre-existing/predicted, 1 newly discovered by this review). Raw failures stay visible below and in the raw outputs.
- **Review-lane maintenance (declared):** the battery as committed at `f7966e6` did not compile. Two mechanical defects fixed by the gate owner before it could judge anything: `[Symbol; 6]::dedup()` does not resolve (this toolchain keeps `dedup` on `Vec` only) → `sorted.to_vec()`; and a moved-value error (`keyed_x` used after move) → clone at first use. No assertion semantics changed.

## 2. Commands executed and observed

```
rch exec -- cargo test -p fsym-core --test symbol_identity_review --locked
  → 3 passed, 2 failed (both are the battery's designed finding tests; see §4)
PYTHONPATH=python .venv-conformance/bin/python -m unittest discover \
  -s python/tests -p "test_lowering_contract_review.py"
  → LoweringContractTests 6/6 PASS (must-hold obligations);
    UntestedOperationIdentityFindings: 3 finding tests, 10 observation failures (see §4)
PYTHONPATH=python .venv-conformance/bin/python -m unittest discover \
  -s python/tests -p "test_surface.py"
  → 139 tests, 8 failures — exactly the 8 clean-HEAD baseline failures
    (euler/finite-differences, extended solvers/transforms, mixed-denominator
    decomposition, solve_linear symbol selection, 4x solve numeric-constant
    output contracts). No new surface failure introduced.
./scripts/build_python_extension.sh → built + smoke-loaded python/fsym_python.so
rch exec -- cargo test --workspace --locked
  → green except the same 2 designed finding tests in symbol_identity_review
rch exec -- cargo clippy --workspace --all-targets --locked -- -D warnings
  → exit 0 (2 pre-existing warnings inside the asupersync path dependency,
    outside -D warnings scope for non-members)
rustfmt --check on changed files → clean; ./scripts/check.sh registries → ok;
br dep cycles → empty
```

## 3. Must-hold obligations (all green)

Same printed text with distinct declared identities never merges (Rust ord/eq + Python `free_symbols`/differentiation/substitution); lifting returns the very declared surface objects; native differentiation refuses a printed name and requires a typed binding; custom `Function`/`Symbol` subclasses are refused before their overrides run; diff receipts replay and every tampered field (rule, rhs, derivative text) is refused; plain symbols keep their historical preimage and cross-architecture golden; assumption-bearing symbols survive pickle (implementation fixture).

## 4. Findings (recorded, with owners; raw failures preserved)

- **F1 (predicted) — `cmp_add_args` compares rendered text.** Unequal atoms that print identically (`f(x)` with distinct identities) compare `Equal` as Add arguments. Minimized: `assert_ne!(cmp, Ordering::Equal)` fails at `symbol_identity_review.rs:221`. Owner: new bead fra-rc-cmp-order-defect.
- **F2 (predicted) — canonical Add order is not a function of the term multiset.** The same multiset inserted in two orders yields two digests (`f01b3f5d…` vs `97a0c07c…`) at `symbol_identity_review.rs:249`. Owner: fra-rc-cmp-order-defect.
- **F3 (predicted) — bounded lift registry evicts the oldest declaration.** After 4096 distinct declared symbols, lift of the first declaration refuses (`NotImplementedError: native result carries a symbol identity with no registered surface object`); `free_symbols` returns `[]`. Owner: new bead fra-rc-lift-registry-bound.
- **F4 (predicted) — name-keyed lanes (`integrate`/`series`) return undeclared lookalikes** instead of the declared atom; declared non-goal of this slice, remains documented debt. Owner: carried by WS18/WS19 residual slices' lane obligations.
- **F5 (NEW, found by this review) — equivalent assumption spellings split into distinct atoms.** `Symbol("x", positive=1)` vs `Symbol("x", positive=True)` produce different `SymbolIdentity` digests (equality/hash/diff/substitution observations all fail; pinned oracle SymPy 1.14.0 treats them as one symbol). The typed identity contract says the digest covers "canonically ordered assumption facts"; spelling-level (not value-level) fact canonicalization breaks that contract. Owner: new bead fra-rc-assumption-spelling.

## 5. Disposition

- `fra-rc-lowering-8w3`: gate artifacts pass → **closes** with this evidence.
- `fra-rc-lowering-gate-ba5`: **closes** as PASS-with-findings; findings are owned by the three new beads above, which depend on this gate.
- No claim in `registries/claims.toml` is promoted; this gate does not close `gate://ws04-semantic-universe` or any workstream.
