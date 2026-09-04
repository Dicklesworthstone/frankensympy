# Golden quarantine note — sympy-1.14.0-cpython (r1)

Recorded 2026-09-04 during fra-shell-drift-zero-collapse-kvg closure.

## Orphan golden files (NOT deleted — removal requires explicit owner authorization per AGENTS.md §4)

These `*.ndjson.ndjson` files exist beside the canonical r1 goldens. Only
`<stem>.ndjson` is authoritative per `golden_name_for()`
(tools/conformance-lab/capture.py); the `.ndjson.ndjson` files are stale
duplicates from an older capture generation:

| File | sha256 | Divergence |
|---|---|---|
| artifacts/conformance/sympy-1.14.0-cpython/goldens/seed_core_atoms.ndjson.ndjson | 38a5e143264f2a92889a91ad07dd4e7a892c1e0168020963978ba5af931cc168 | byte-identical to canonical r1 seed_core_atoms.ndjson (harmless duplicate) |
| artifacts/conformance/sympy-1.14.0-cpython/goldens/seed_function_subclass.ndjson.ndjson | 1d342325e594d4ed77a8775b6dba15510a500a313e7efa275f5f15e652a22370 | DIVERGES from canonical r1 seed_function_subclass.ndjson (11f1bfde52aa5517886c3e985fab2cbc36a29e486c631f1515276430c7d05939) — stale TypeError-era capture of the subclass fixture |
| artifacts/conformance/sympy-1.14.0-cpython/goldens/seed_held_forms.ndjson.ndjson | 4688f2bcefce2ff4518471848b4c0d5b7733ab590d96c9f5fd13760f30c9bd04 | byte-identical to canonical r1 seed_held_forms.ndjson (harmless duplicate) |

## Profile revision r1 → r2 (refresh rationale)

The r1 goldens were captured 2026-08-23. Afterwards the seed fixture
`seed_function_subclass.json` gained the `eval_zero_collapse` semantics
(oracle_runner.py / candidate_runner.py `eval` hooks), so the r1 golden for
`subclass/ConstitutiveLawZero_zero_collapse` recorded the applied form while
both live interpreters now collapse `ConstitutiveLawZero(0, k)` → `S.Zero`.
Per the compatibility contract (changing fixture semantics creates a new
profile; goldens are immutable per profile), revision **r2**
(`profiles/sympy-1.14.0-cpython-r2.toml`, profile_id
`sympy-1.14.0-cpython-r2`) was created and freshly captured on 2026-09-04
against the pinned oracle (sympy 1.14.0 @ 16fa855354eb7bcabd3fe10993841e03b1382692,
CPython 3.14.4, /home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0).

r1 goldens and run manifests are preserved untouched as immutable history.
Candidate-vs-r2 diff: 14/14 admitted, 0 discrepancies (exit 0).
