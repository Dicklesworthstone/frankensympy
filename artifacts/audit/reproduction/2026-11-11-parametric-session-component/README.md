# Exact-source Session component reproduction

This is scoped functional and compiler evidence for the unchanged FrankenSymPy root library, native facade, Session and NDJSON CLI. It is **not** a successful build of the full production Cargo dependency graph or an asynchronous runtime/profile gate. Four attempts to compile that full graph ended when the `asupersync` compiler was terminated by SIGKILL; the kernel cause was not independently established. This component build composes the required whole source modules without changing any function.

## Recorded result

- Repository source commit: `e222f96c7008c3a81564541758d1f82b9c7a40f4`.
- Pinned donor commit: `b3e08ac0bd773a5bbdd4f3c2f4624d550352ec0f`.
- Actual root `src/lib.rs` and `src/bin/frankensympy.rs` are Cargo target paths; neither is copied or modified.
- **28 tests passed**: native parametric facade 10, typed Session 10, fresh-process CLI 4, optimizer integration adversaries 4.
- Strict Clippy passed for the unchanged root library, binary and all four integration test targets using explicit package selection and `--no-deps`.
- All 455 recorded original input file hashes and harness input hashes stayed equal across test and Clippy snapshots. All 46 registry package identities/checksums match the original repository lock. Twenty dependency packages use complete original path sources.

The runtime component imports whole original `workspace`, `protocol`, `fmap`, `publication` and `repair` modules. The pinned codec component imports 14 whole original modules and its original systematic index table. Wrapper files contain only module wiring, reexports and crate attributes. No functions are replaced, copied, shortened or stubbed. The same relevant default donor features are enabled. Original dependency unit tests remain disabled as they are in a normal dependency build.

`source-inventory.json` records every selected source hash, selected modules, exact dependency identities, and counts/digests for the 12 omitted FrankenSymPy runtime and 1535 omitted donor source files. `reproduce.py` regenerates both omission lists from the pinned trees and verifies their digests. The donor build script only exports revision metadata unused by the selected modules; its omission and omitted crate-root lint context are recorded explicitly.

## Reproduction

These files preserve the original local absolute and relative paths for provenance. **Relocate into a new scratch directory before building; do not run Cargo in this committed evidence directory.** Obtain the repository/donor commits above and install the pinned `nightly-2026-08-20` toolchain. The original external registry dependencies must already be available for offline reproduction.

```sh
python reproduce.py --repository /path/to/frankensympy --donor /path/to/asupersync --output /path/to/new-component-scratch
cd /path/to/new-component-scratch
export CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
cargo +nightly-2026-08-20 metadata --offline --locked --filter-platform x86_64-unknown-linux-gnu --format-version 1 > metadata.json
python record_scope.py pre-run-scope.json
cargo +nightly-2026-08-20 test --offline --locked --test native_parametric --test parametric_session --test parametric_cli --test optimization_integration_adversaries -j1
cargo +nightly-2026-08-20 clippy --offline --locked -p frankensympy-session-source-component-check --no-deps --lib --bin frankensympy --test native_parametric --test parametric_session --test parametric_cli --test optimization_integration_adversaries -j1 -- -D warnings
python record_scope.py post-run-scope.json
```

Use an appropriate external `CARGO_TARGET_DIR`, `RUSTUP_HOME` and `CARGO_HOME` if required by the host. No original source or dependency tree is included here, and no target binaries/cache are preserved. The Linux metadata filter avoids fetching unused target dependencies; it does not replace a dependency or alter the Linux compiled graph.

## Logs and scope

`component-tests.txt` captures all 28 test outcomes. `root-component-clippy.txt` is the successful root-only strict pass. Seven warnings about unused original donor bootstrap entries/nightly attributes remain visible and were not suppressed. `component-clippy.txt` preserves the first failed command, which also applied strict lints to the scratch workspace's original donor module members and stopped at 30 donor diagnostics before root validation. Explicit package selection with `--no-deps` scopes the successful pass to root targets; it does not establish donor lint cleanliness.

`input-snapshots.json` preserves original pre/post snapshot digests and equal source/harness hash-map digests; one shared selected-source inventory avoids repeating the large module lists. `validation-result.json` records commands, environment and limitations. Reproduction creates fresh full scope snapshots including the regenerated omitted-file lists. A passing component run does not close full-workspace, asynchronous cancellation/scheduling, durability/transport or compatibility certification gates.
