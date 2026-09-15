# WS01 conformance-laboratory residual statement (`fra-ps0`)

## 1. Record

- **Bead:** `fra-ps0` (WS01 residual, P2), authored and executed by CrimsonTurtle, verified 2026-09-15 against the working tree at `main` (post `dfb7e7c`).
- **Inputs:** `docs/WORKSTREAM_GRAPH.md` §6 (WS01 deliverables/acceptance/forbidden), `docs/CONFORMANCE_AND_BENCHMARKING.md`, `tools/conformance-lab/` (file-inspected), live gate runs (below).
- **Not claimed:** WS01 closure, `gate://ws01-conformance-lab` closure, any compatibility claim. `lab-corpus` reports `certified: false` by construction and the moving-head lane refuses `--certify`.

## 2. Deliverable inventory (WORKSTREAM_GRAPH §6) — verified against files, live

| WS01 deliverable | Evidence (file) | Status |
|---|---|---|
| Immutable SymPy 1.14.0 environment manifest | `profiles/sympy-1.14.0-cpython*.toml`; oracle runner rejects unpinned `PYTHONHASHSEED` (`capture.py:1949`) | supported |
| Separate-process fixture protocol | `capture.py` subprocess lanes; candidate/oracle runners refuse unpinned seed (`candidate_runner.py:40-52`) | supported |
| Source/reflection inventory generator | `inventory_runner.py` records modules, `__all__`, class records, signatures | supported |
| Observation envelope + comparator registry | 11-field envelope (`capture.py:149-161`); `comparators.py`; gate comparator `construction_only` | supported (comparator registry breadth = residual R2) |
| Mismatch minimizer + discrepancy writer | `minimize.py`; `corpus_gate.py` ledger ("ledgered-open … FIX them, never weaken the comparator") | supported |
| Moving-head drift lane | non-certifying by design; gate output `broken_candidate_rejected_by_construction_only: true` | supported |
| Exact environment/build capture | `environment_records.py` | supported |

## 3. Acceptance properties (WORKSTREAM_GRAPH §6)

| Property | Evidence | Status |
|---|---|---|
| oracle/candidate cannot import/share one another | `./scripts/check.sh lab` isolation checks, 123 tests `OK` | pass |
| fixtures reproduce across fresh processes | pickle dumps round-trip through fresh `python -P -s` subprocesses per fixture (`capture.py:895+`); corpus gate `unledgered: 0` | pass |
| deliberate mismatches detected | broken-candidate rejection; assumption contrast pairs in generated corpus (`generate_corpus.py:103+`); adversarial/custom-subclass corpus (`build_adversarial`) | pass |
| comparator mutation fixtures fail | registered mutants (`mutants.json`, `test_registered_mutants.py`) | pass |
| moving-head cannot certify | enforced by the lab gate | pass |

## 4. Remaining obligations, each owned

- **R1 — hash-seed-1 observation sweep.** The corpus sweep runs at the profile's pinned seed only; profile-declared `hash_seed` validation exists (`capture.py:268`) but no registered lane captures candidate observations at seed 1 against a fresh same-seed oracle capture. **Owner: `fra-rc-surface-nvv`** (open; its steps (a)+(b) are exactly this).
- **R2 — observation-comparator wiring.** The corpus gate's comparator is `construction_only`; the exact-surface observation comparator is not yet wired as a gate comparator, so observation drift beyond construction is unmeasured until R1's sweep exists. **Owner: `fra-rc-surface-gate-0eh`** (open; step (c) depends on R1's sweep).
- **R3 — restore-path serialization adversaries.** The corpus carries adversarial fixtures and pickle record validation detects mismatch/drift, but no registered sweep feeds malformed or hostile pickles to the fresh-process `pickle_loader` trust boundary. **Owner: new bounded bead `fra-toi`** (created by this statement; acceptance commands included there).
- **R4 — stale claim in the 2026-09-12 measured baseline:** the statement "fresh pickle processes is not covered by any registry-declared sweep" no longer holds — per-fixture subprocess round-trips are wired in `capture.py`. Recorded here so the baseline correction is visible; `fra-rc-surface-nvv` steps (a)–(c) remain the live work.

## 5. Acceptance commands (executed for this statement)

```
./scripts/check.sh lab        → 123 tests OK; isolation, self-test, suite-smoke, moving-head refuse-certify
./scripts/check.sh lab-corpus → deterministic corpus (sha 13a4ceb04a78), construction_only,
                                unledgered: 0, broken candidate rejected
./scripts/check.sh registries → ok
br dep cycles                 → empty
br list: WS01 orphan          → none (fra-ps0 closes with this statement; R1/R2 owned by
                                open beads, R3 by new fra-toi)
bv --robot-triage             → parses, source_authority complete
```

No golden was regenerated, no comparator weakened, no closure asserted by prose: `certified: false` stands on every lane.
