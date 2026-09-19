# fra-g4w solver/surface correction slice — TurquoiseHorizon 2026-09-19

## Root cause of the recorded 8-failure surface baseline

The exact Add lane collected pure numeric constants but never collected
like terms, and Mul with an exact rational coefficient never distributed
over a single Add factor. Upstream SymPy 1.14.0 does both in
Add.flatten/Mul.flatten, so `x - x` constructs 0 at the oracle while the
shell held `Add(x, -x)` — poisoning solve classification, solve_linear's
coefficient extraction (a coefficient like `m + z - (m + z) + 1`), and
the mixed-denominator contracts.

## Fix (crates/fsym-core/src/lib.rs, canonical layer)

- `collect_like_terms` (replacing fuse_numeric_terms): splits each exact
  term into (rational coefficient, base), sums coefficients per base,
  drops zero sums; pole-bearing bases are exempt so oo - oo semantics are
  never invented; constants fuse as before.
- `canonicalize_mul_args`: an exact rational coefficient distributes over
  a single Add factor (upstream Number*Add), and `wrap_mul_factors`
  collapses one-factor Muls.
- `fsym-solvers`: exact rational quadratic roots enumerate ascending
  (pinned oracle order), in both solve_poly and solve_quadratic lanes.

Two unit tests that pinned the OLD (+/-) root order were aligned to the
pinned oracle outputs (verified live: solve(x^2-5x+6)==[2,3],
solve(x^2-x-6)==[-2,3], solve(4x^2-1)==[-1/2,1/2]). The deep-chain test's
depth pin was updated to the post-distribution contract (telescoping flat
Add), preserving its purpose: iterative construction without stack
growth.

## Verification

- Python surface suite: 141 tests OK (was 141 tests / 8 failures).
  Full python discovery: 158 tests, only the 3 name-keyed integrate/series
  subtests fail — the recorded finding carried by WS18/WS19 residuals.
- Oracle differential (same expressions through pinned SymPy 1.14.0 and
  the shell): all contracts identical (differential_oracle.txt vs
  differential_shell.txt; only the pre-existing internal Dummy printing
  convention differs, which the suite pins separately).
- Rust workspace: fmt --check exit 0; clippy --workspace --all-targets
  --locked -- -D warnings exit 0; cargo test --workspace --locked exit 0
  (90 suites ok; transcript workspace_test.txt). UBS on the two changed
  files: 0 critical.
- Quality chain executed locally; remote attempts today were blocked by
  the fleet-side planner failures recorded in
  artifacts/audit/ws15_durable_m5e_th1/QUALITY_CHAIN.txt.

## Claim effects

Surface solver contracts restored to oracle parity. No completeness
certificates, no profile certification, no performance claims. The bead
stays in_progress pending the independent gate review its contract
requires (implementer = this session; reviewer must differ).
