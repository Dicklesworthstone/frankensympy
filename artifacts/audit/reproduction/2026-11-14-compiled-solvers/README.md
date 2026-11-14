# Compiled Session solver validation

The progress artifact one directory above records commands, terminal results,
toolchain, pinned donors, source inventories and the normal build failure.
`component-integrated-tests.txt` contains the passing 71-test run after concurrent
main commit `0162ce3`; the baseline transcript is retained separately. These are
two executions of the same 71 cases, not 142 different tests.

The component compiles the complete actual root library, CLI and selected tests,
25 complete original path packages, five whole synchronous runtime modules and
fourteen whole pinned codec modules. The real numeric consumer, FrankenNumPy
and FrankenSciPy dependency closure and features remain active. No source
function is replaced. All 671 source hashes and harness hashes match in the
recorded pre-run/post-test inventories. All 58 registry identities/checksums
match the production lockfile.

The retained manifests, wrappers and recorder are the exact executed scratch
composition. Their absolute paths record the original environment. To recreate
it elsewhere, copy this directory outside the repository, substitute only the
repository/donor/scratch path prefixes, retain the original pinned source trees,
and regenerate `metadata.json` with locked Cargo metadata before running
`record_scope.py`. Use the pinned `nightly-2026-08-20` toolchain and the environment
and explicit ten-target command in the progress artifact. Do not substitute
runtime methods or numerical kernels to make the composition compile.

The component omits the asynchronous runtime and transport graph. It therefore
does not establish full-workspace, asynchronous cancellation, performance,
compatibility-profile or campaign success. The normal production build received
SIGKILL while compiling the pinned `asupersync` library and ran no tests; its
original log is retained. Broader gates remain open.
