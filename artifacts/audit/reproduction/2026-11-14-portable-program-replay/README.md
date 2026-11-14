# Portable compiled-program replay: scoped validation

Recorded 2026-11-14. This preserves actual test outcomes and a reconstruction recipe after the execution environment disconnected.

## Observed results

- Original locked Cargo workspace: 11 source-admission tests and 10 candidate-codec tests passed. The complete observed native terminal output is retained in `native-tests.txt`.
- Unchanged-source Session/CLI component: all 57 tests in the eight targets below passed with terminal exit 0. The actual root library, CLI, tests and mathematical crates were compiled.
- The first component compile exposed an incomplete exhaustive match in the existing artifact-transfer test helper. Adding explicit exports for `BoxOptimizationSearch`, `CompiledResidualSystem` and `CompiledObjective` preserved every existing case and assertion.
- The normal full-production-graph test command was intentionally interrupted with exit 130 while additional runtime dependencies compiled. It reached no selected tests and is not a mathematical failure or an observed asupersync SIGKILL.
- Scoped component Clippy started, but its terminal result was not observed before disconnection. Native scoped Clippy did not start.

Before executor loss, all 533 pre-run/post-test original-source hashes matched. All 46 registry package identities/checksums matched the repository lockfile. The component used 20 complete original path packages, five whole runtime modules and fourteen whole pinned donor modules plus the original table. It inventoried twelve runtime and 1,535 donor files as omitted. No source function was substituted.

The root raw logs and complete hash/module inventories remained in scratch when the executor disconnected and are not claimed to be preserved here. The main progress JSON records confirmed terminal outcomes, scope and exact original commands. The original native raw log was already committed before the loss.

## Reconstructed setup recipe

`prepare_component.py` was reconstructed after executor loss from the existing tracked whole-module templates and observed setup. Root independently checked its template anchors through GitHub. This reconstructed recipe was not executed in this session and does not retroactively reproduce the lost logs or hash list.

Use a checkout containing the test-helper fix with blob `303cb43587939f731193b139270b587a53fabfde`. Prepare the exact pinned donors with the repository's existing donor-fetch instructions. The script requires asupersync at `b3e08ac0bd773a5bbdd4f3c2f4624d550352ec0f` and refuses changed registry identities/checksums. Choose a new output directory outside both source trees.

From the repository root:

```bash
REPLAY_REPOSITORY="$PWD"
python artifacts/audit/reproduction/2026-11-14-portable-program-replay/prepare_component.py \
  --repository "$REPLAY_REPOSITORY" \
  --donor ../asupersync \
  --output ../frankensympy-replay-component
cd ../frankensympy-replay-component
export CARGO_BUILD_JOBS=1
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_TARGET_DIR="$REPLAY_REPOSITORY/target"
```

Use the repository-pinned `nightly-2026-08-20` compiler. The original successful invocation used that toolchain through its task-local rustup environment; the commands below select it explicitly. Offline execution requires the locked registry dependencies to be cached.

```bash
cargo +nightly-2026-08-20 metadata --offline --locked \
  --filter-platform x86_64-unknown-linux-gnu --format-version=1 > metadata.json
python record_scope.py component-pre-run.json
cargo +nightly-2026-08-20 test --offline --locked \
  -p frankensympy-session-source-component-check \
  --test compiled_program_transport --test compiled_program_cli \
  --test compiled_artifacts --test compiled_residual_session \
  --test compiled_residual_products --test compiled_objective_session \
  --test artifact_transfer --test artifact_transfer_session \
  -- --test-threads=1
python record_scope.py component-post-tests.json
```

The recorder regenerates selected hashes, omitted-module lists, package identities/features and harness hashes for the new run. Compare its source maps before and after execution. Declaring all current root integration targets mirrors the observed preparation; only the explicit eight targets above are executed.

## Limits

This component imports whole unchanged `workspace`, `protocol`, `fmap`, `publication` and `repair` runtime modules and the selected pinned codec/support files. It omits the remaining asynchronous runtime and transport graph. A component pass does not establish full-workspace, asynchronous behavior, compatibility-profile, performance or complete campaign success. The production CI gates remain responsible for that wider scope.
