# Local-first release and validation

**Status:** normative architecture contract  
**Scope:** canonical local gates, Doodlestein orchestration, evidence artifacts, release state machine, machine matrix, GitHub workflow role, signing, reproducibility, and rollback

## 1. Authority

The canonical build, test, benchmark, packaging, signing, and publication gates execute on controlled machines through `Dicklesworthstone/doodlestein_self_releaser` or an equivalent local orchestrator.

GitHub Actions workflow files may document jobs, receive dispatches, or be executed by local runners. A green GitHub-hosted check is not the authoritative release receipt and is never the only gate.

## 2. One local entry point

`scripts/check.sh` is the human and orchestrator entry point. Named profiles compose exact commands rather than duplicating logic in YAML.

Planned profiles:

- `format`;
- `metadata`;
- `registries`;
- `portable-verifiers`;
- `unit`;
- `conformance`;
- `lab`;
- `fuzz-smoke`;
- `bench-smoke`;
- `package`;
- `release-candidate`;
- `all`.

Each profile emits a canonical manifest and separate host telemetry.

## 3. Evidence artifact

Every gate emits:

- schema and gate ID;
- source commit/tree;
- dirty-state refusal;
- exact toolchain and lock digest;
- target, features, and environment profile;
- command list and exit status;
- semantic result counts;
- input/corpus roots;
- produced artifact roots;
- start/end logical sequence;
- machine class and architecture;
- host telemetry separated from canonical semantic fields;
- signer identity for release gates.

A textual log is useful diagnostics, not the receipt itself.

## 4. Release state machine

```text
Draft
 -> LocallyValidated
 -> MatrixValidated
 -> ReproducibilityCompared
 -> Packaged
 -> Signed
 -> Staged
 -> Published
 -> Promoted

Any state -> Quarantined
Published/Promoted -> Superseded | Revoked
```

Transitions require the registered evidence set. Publication cannot infer earlier states from artifact presence.

## 5. Machine matrix

Initial controlled matrix:

- Apple Silicon current macOS, optimized native build;
- x86-64 Linux with AVX2 baseline;
- high-core-count AMD x86-64 optimized build;
- portable x86-64 feature floor;
- Wasm target for portable verifier and selected core crates;
- Python profile matrix declared separately.

Performance claims name the exact machine class. Correctness gates run on multiple thread counts and deterministic lab schedules.

## 6. Doodlestein handoff

The repository supplies machine-readable release metadata:

- gate graph;
- commands;
- required machine classes;
- artifacts and expected locations;
- signing policy;
- promotion rules;
- rollback/revocation rules;
- concurrency and resource requirements.

Doodlestein owns orchestration mechanics. FrankenSymPy owns semantic gate definitions and validates the resulting evidence before accepting a release state transition.

## 7. GitHub workflow role

Workflow YAML is a projection of local gates. It must:

- call repository scripts rather than reimplement tests;
- declare that local evidence is canonical;
- avoid secrets or publication authority not available to the local release flow;
- remain useful for mirrors and contributors;
- never weaken a gate because hosted resources are constrained.

A workflow skipped, queued, unavailable, or executed by GitHub does not change the local release state.

## 8. Reproducibility

For a release candidate:

- build at least twice from clean source on the same machine class;
- compare canonical artifacts byte-for-byte where claimed;
- compare semantic manifests where platform-bound bytes legitimately differ;
- record nondeterministic sections explicitly;
- refuse reproducibility wording outside the certified matrix;
- preserve source, lock, toolchain, environment, and command closure.

## 9. Performance gates

Benchmarks run on quiet controlled machines with:

- warmup and measurement plan;
- same-invocation old/new or incumbent/native arms;
- A/A control;
- load, thermal, frequency, and memory telemetry;
- exact corpus and route IDs;
- statistical method and stop rule;
- semantic equivalence gate before timing;
- archived raw samples.

No release is blocked by an uncalibrated noisy microbenchmark. No performance claim is promoted without its registered evidence.

## 10. Signing and provenance

Release publication includes:

- signed source/tree identity;
- signed gate-manifest root;
- SBOM and dependency registry root;
- toolchain fingerprint;
- package/wheel/crate artifact digests;
- reproducibility comparison;
- compatibility and claim matrices;
- revocation channel and key epoch.

Signatures attest origin and gate completion, not mathematical truth.

## 11. Recovery and rollback

The release process is resumable and idempotent. Partial upload or cancellation cannot produce a promoted release.

- immutable artifacts upload before mutable channel pointers;
- staging verifies every digest;
- promotion updates one atomic manifest/channel root;
- interrupted promotion is recovered from durable intent/complete markers;
- rollback points to a prior signed manifest rather than mutating old artifacts;
- revocation is append-only and preserves audit history.

## 12. Required local blockers

A release is blocked by:

- dirty source tree;
- toolchain or dependency registry mismatch;
- unsafe/dependency violation;
- registry/DAG inconsistency;
- portable verifier closure violation;
- conformance or certificate mutation failure;
- documentation claim inflation;
- missing matrix receipts;
- missing reproducibility comparison where claimed;
- unsigned or mismatched artifacts;
- GitHub-only evidence with no canonical local receipt.
