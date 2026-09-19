# fra-j3y closure evidence — TurquoiseHorizon 2026-09-19

Implementation under review: e2887d7 (author BoldGorge; reviewer independent).

## Acceptance commands and results

1. Finding test: `python/tests/test_lowering_contract_review.py::LoweringContractTests::test_finding_equivalent_assumption_spellings_become_distinct_atoms`
   → ok under `.venv-conformance` (5/5 observations: equality, hash,
   differentiation across spellings, substitution across spellings,
   arbitrary fact spelling `foo=1` vs `foo=True`).
   Full suite: 9 tests; 6 ok; the 3 FAILs are the recorded name-keyed
   integrate/series lane finding, explicitly carried by the WS18/WS19
   residual beads (fra-a5x / fra-7kf) in the fra-rc-lowering-gate-ba5
   closure record — not a must-hold regression of this bead.
   All LoweringContractTests must-holds green.
2. `cargo test -p fsym-core -p fsym-assumptions --locked`:
   - Remote attempt vmi1156319: 128 tests ok except
     fresh_process_id_stability::canonical_terms_match_golden_specification,
     which failed ONLY because the worker's synced tree lacked the
     committed runtime-read fixture
     artifacts/conformance/fixtures/deterministic_term_identity_v1.json
     (rch materialization gap for non-Rust runtime-read files; transcript
     test_core_assumptions_remote_worker.txt).
   - Local execution (fleet degraded: E412 planner fail-open partial sync on
     vmi1152480): exit 0, all suites ok, fixture present
     (test_core_assumptions_060.txt).

## Lockfile/registry repair included in this closure

The shared sibling checkout ../asupersync advanced 0.5.0 -> 0.6.0
(b3e08ac0b, 2026-09-18 22:33 EDT) after this bead's earlier green runs,
breaking every `--locked` command workspace-wide (asupersync is a path
dependency; Cargo.lock pinned 0.5.0). Repair: minimal offline re-resolution
(5 lock entries: asupersync, asupersync-macros, franken-decision,
franken-evidence, franken-kernel). Verified on 0.6.0: cargo check
--workspace --all-targets --locked exit 0; the acceptance suites above pass
on 0.6.0. registries/dependencies.toml asupersync commit pin updated
3cb2dc6d... -> b3e08ac0b... with provenance comment.
