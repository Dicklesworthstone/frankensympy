# AGENTS.md — FrankenSymPy

> Binding repository guidance for coding and research agents.

---

## 0. Owner instructions take precedence

Follow the repository owner's explicit task instructions. Do not reinterpret a concrete request into a safer-looking but materially different task, do not claim work was performed when it was not, and do not repeat a question whose answer is already available.

---

## 1. Read the architecture before changing code

Before substantive work, read the relevant parts of:

1. [`COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md`](COMPREHENSIVE_PLAN_FOR_FRANKENSYMPY.md)
2. [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md)
3. [`docs/SOURCE_PROJECT_AUDIT.md`](docs/SOURCE_PROJECT_AUDIT.md)
4. [`docs/WORKSTREAM_GRAPH.md`](docs/WORKSTREAM_GRAPH.md)
5. [`registries/workstreams.toml`](registries/workstreams.toml)
6. [`registries/claims.toml`](registries/claims.toml)
7. the subsystem-specific contract under `docs/`

The comprehensive plan is the architectural overview. The constitution contains non-negotiable rules. Machine-readable registries are authoritative for status, IDs, and dependency edges.

If implementation pressure conflicts with a constitutional rule, stop and surface the conflict. Do not silently weaken the rule, comparator, verifier, profile, or gate.

---

## 2. Current repository reality

The repository currently contains a skeleton workspace and a detailed public architecture/implementation package. It is **not** yet a working or certified SymPy replacement.

Agents must not claim that the repository currently has:

- complete or partial drop-in compatibility unless the exact profile gate says so;
- a live native symbolic engine merely because a crate/file/API name exists;
- proof-producing algorithms without a live independent verifier and mutation gate;
- RaptorQ repair without end-to-end encode/loss/decode/digest/schema/resume evidence;
- conformal/e-process monitoring without the registered monitor and validation artifacts;
- performance wins without same-invocation live-incumbent and semantic-admission evidence;
- Franken-suite integrations without end-to-end adapter gates.

Capability status lives in [`registries/claims.toml`](registries/claims.toml). Present-tense prose must agree with it.

---

## 3. Git: `main` is authoritative

All requested work in this repository lands on `main` unless the owner explicitly says otherwise.

- Do not push work only to a feature branch and describe it as landed.
- Do not open a PR when the owner explicitly requested direct commits.
- Do not write to or cite `master` as the active development branch.
- Do not claim a push until the remote commit is independently visible.

### Mandatory remote verification

After every push or connector write:

1. capture the returned commit SHA;
2. fetch that exact remote commit through GitHub or `git fetch`/`git show`;
3. verify the expected message and diff/file path;
4. periodically fetch the remote `main` head;
5. record the actual SHAs in the final report.

A local commit, an API success response without follow-up, or an intended command is not proof that GitHub contains the work.

For long planning or implementation tasks, use a large number of coherent incremental commits. Each commit should preserve a readable, buildable or documentation-consistent state. Commit-stream pumping—empty, cosmetic, or artificially split commits—is forbidden.

---

## 4. No deletion or destructive cleanup

**Never delete a file or directory without clear, written owner permission.** This includes files created during the current session.

Forbidden without exact explicit authorization:

- `git reset --hard`
- `git clean -fd` or stronger variants
- `rm -rf`
- destructive checkout/restore commands that overwrite work
- force pushes
- history rewrites
- deleting branches, tags, releases, files, artifacts, or persistent data

When rollback is needed, prefer additive fixes, revert commits, backups, or a new corrective commit. If a destructive command is explicitly authorized, restate the exact command and affected paths before execution and record the authorization and outcome.

---

## 5. Suite-wide reward-hacking rules

When the suite-wide `/data/projects/AGENTS.md` is available, its reward-hacking and work-graph rules apply here. The following are always forbidden:

- gate self-weakening;
- proof-class inflation;
- golden regeneration reflex;
- commit-stream pumping;
- tautological tests;
- easy-lever or workload cherry-picking;
- close-pump abuse;
- scope-splitting to evade a blocked requirement;
- spec editing presented as implementation progress;
- conformance metastasis used to avoid fixing central behavior;
- dependency smuggling;
- benchmark-path hard-coding.

A self-speedup against an older FrankenSymPy build is maintenance evidence, not a competitive win. A performance win requires the live incumbent in the same invocation or a rigorously controlled paired run, with semantic admission first.

Reporting a loss or a blocked hypothesis is successful engineering. Do not alter the gate or narrative to manufacture a win.

---

## 6. Compatibility is a core requirement

The old placeholder statement that “backwards compatibility does not matter” is not applicable to SymPy-facing behavior.

Distinguish two things:

### Public compatibility profile

Once a SymPy profile is frozen, its observable behavior is an immutable target. Do not break or redefine it for implementation convenience. Profile changes require a new immutable profile and differential evidence.

### Internal pre-1.0 native APIs

Internal Rust APIs, private schemas, and implementation structure may evolve before 1.0 when doing so removes technical debt. Do not add compatibility shims for unreleased internal APIs unless a persisted artifact, cross-crate consumer, migration plan, or explicit owner instruction requires them.

Never use “we are early” to excuse wrong Python class identity, `evaluate=False`, assumptions, hashing, printing, pickle, warning, exception, mutation, or module behavior.

---

## 7. Core architectural invariants

### 7.1 Dual-lane object model

- Python shell owns Python-visible identity and extensibility.
- Native kernel owns deterministic mathematical semantics.
- Arbitrary Python subclasses remain ordinary Python classes.
- Exact-class and override checks guard native fast paths.
- Unknown nodes remain conservative opaque nodes.
- Upstream SymPy is never a certified production fallback.

### 7.2 Three representations

Keep separate:

- Surface Object Graph;
- Semantic Term DAG;
- Derivation Evidence Graph.

Do not use printed strings as IR. Do not canonicalize away held/custom surface behavior. Do not treat a search trace or provenance volume as proof.

### 7.3 Stable identity

- Use typed IDs, never generic strings internally where kind matters.
- `TermId` is not Python hash, arena handle, pointer, row ID, or graph vertex.
- Stable identities exclude scheduling, time, memory address, and cache state.
- Validate canonical payload at trust boundaries; do not trust digest equality alone.

### 7.4 Domains and assumptions

- Domain, sort, assumptions context, branch policy, and compatibility facts are distinct.
- `Unknown` never becomes true or false.
- Contradictory contexts do not prove everything by default.
- Conditional rewrites require discharged side conditions or remain guarded.
- Cache keys include all identity-relevant context/profile/rule/precision inputs.

### 7.5 Evidence

Values and evidence classes are separate. Do not promote:

- heuristic candidate → proof;
- selector posterior → certificate;
- e-process/conformal score → mathematical evidence;
- oracle parity → mathematical truth;
- sampled numerical agreement → exact identity;
- worker vote/signature/reputation → proof;
- RaptorQ decode → integrity/authenticity/schema validity/proof;
- stored `verified=true` flag → accepted result.

### 7.6 Generator/verifier separation

- Optimizing generator crates cannot be dependencies of their independent verifier crates.
- Verifiers check the exact typed claim and no stronger claim.
- High-value optimized verifiers retain a simple reference lane.
- Every certificate family needs negative, adversarial, and mutation tests.

### 7.7 Structured execution

- asupersync is the only async/concurrency runtime.
- Every controlled spawned task has one owning region.
- No detached background work.
- Cancellation is request → drain → finalize.
- Candidate publication and accepted publication are separate phases.
- Generators cannot consume verifier-reserved budget.
- Fallback never resets accounting.
- Do not claim forced cancellation bounds through arbitrary non-cooperative Python or foreign code.

### 7.8 Persistence and distribution

- Persistence is optional and outside the algebraic hot path.
- Database rows and graph reachability do not define identity or proof.
- Candidate and verified cache namespaces remain separate.
- Checkpoints are typed normalized state, never process memory dumps.
- Remote workers generate candidates; the local coordinator verifies before publication.
- Optional integrations cannot alter core term/proof semantics.

---

## 8. Rust toolchain and memory safety

Use Cargo for Rust workspace management.

- Rust edition: 2024.
- Follow the pinned toolchain file when present.
- Ordinary native crates use `#![forbid(unsafe_code)]`.
- No C/C++ CAS or arbitrary-precision engine through FFI.
- No second async runtime.
- Direct hand-written CPython C-API code is prohibited.
- A contained Python bridge may use an audited safe Rust binding layer such as PyO3 as a foundational dependency exception.

If an optimization eventually requires an unsafe island, it needs a dedicated architecture amendment, isolated crate, safe total API, scalar reference lane, fuzz/property/differential tests, Miri/sanitizer-equivalent gates, architecture CI, and explicit owner approval. Do not add workspace-wide unsafe allowances.

---

## 9. Dependency policy

Prefer:

- Rust standard library;
- asupersync;
- narrow Franken-suite adapters;
- only foundational external crates that pass written admission review.

A dependency proposal must document:

- exact need;
- existing alternatives;
- transitive tree and features;
- unsafe/FFI/build-script/network behavior;
- determinism and serialization impact;
- Wasm/platform impact;
- maintenance and license;
- containment boundary;
- replacement/removal strategy;
- required gates.

Forbidden:

- Tokio and Tokio-dependent stacks;
- `async-std`, `smol`, or another runtime;
- C/C++ CAS/big-number FFI;
- hidden Python/SymPy execution engines;
- framework-heavy agent orchestration dependencies;
- runtime code loaders;
- opaque external solver binaries advertised as verified exact engines.

Do not smuggle a dependency through a dev/build/optional feature to evade review.

---

## 10. Layering

Target layers:

```text
L7  Product packaging, CLI, Python distributions
L6  Protocol, Python bridge, Wasm, generated-code targets
L5  Persistence, distribution, graph index, repair adapters
L4  Planning, portfolios, workspaces, compilation, services
L3  Symbolic algorithm generators
L2  Terms, domains, assumptions, claims, proof kernel, verifiers
L1  Exact arithmetic, canonical encoding, deterministic collections
L0  IDs, budgets, outcomes, schemas, capabilities
```

Higher layers may depend on lower layers. Cycles are forbidden. Core semantic crates cannot depend on Python, persistence, graph, network, or product layers. Verifiers cannot depend on optimizing generators.

Before adding a cross-layer edge, update the dependency registry/architecture and explain why the boundary is correct.

---

## 11. Code editing discipline

### 11.1 Do not use blind bulk rewrites

Do not run brittle regex or search-and-replace scripts over source files. Make semantic edits deliberately. Deterministic code generation from reviewed registries is allowed when generation itself is a designed repository tool with dirty-diff CI.

### 11.2 Do not create variants

Do not create files such as:

- `foo_v2.rs`
- `foo_new.rs`
- `foo_fixed.rs`
- `foo_final.rs`

Revise the authoritative implementation. New files require a real module, trust, dependency, ownership, or schema boundary.

### 11.3 Keep commits coherent

Each commit should do one reviewable architectural or implementation unit. Avoid mixing unrelated refactors, gate changes, generated churn, and feature behavior.

### 11.4 Do not hide failures

Do not swallow errors, silently fall back, return a candidate as accepted, or alter expected outputs to match incorrect code. Preserve minimized counterexamples and discrepancy records.

---

## 12. Work-graph and Beads discipline

The architecture DAG is in [`registries/workstreams.toml`](registries/workstreams.toml). Beads tasks may be created only after the conversion gate in [`docs/WORKSTREAM_GRAPH.md`](docs/WORKSTREAM_GRAPH.md).

Every implementation task needs:

- bounded objective and non-goals;
- exact dependencies;
- owned files/crates/registries;
- immutable input universe;
- deliverable;
- independent gate owner;
- objective acceptance commands;
- unit/property/differential/metamorphic/adversarial obligations;
- benchmark and live-incumbent obligations when relevant;
- claim/discrepancy effects;
- cancellation/resource/failure semantics;
- forbidden shortcuts;
- closure artifacts.

JSONL is authoritative for the executable work graph when Beads is initialized. Derived databases are disposable. Structural graph editing is single-writer and `br dep cycles` must remain empty.

Useful commands when configured:

```bash
br ready
br list --status=open
br show <id>
br update <id> --status=in_progress
br close <id> --reason "Completed with gate artifacts: ..."
br sync --flush-only
br dep cycles
bv --robot-triage
bv --robot-next
```

Do not close a task because code was committed; close it when the named gate artifacts pass.

---

## 13. Conformance discipline

Compatibility work must use the immutable profile and isolated upstream oracle.

Required coverage includes, as applicable:

- source/reflection inventory;
- exact class/metaclass/MRO/signature/module behavior;
- evaluated and held construction;
- custom subclasses, converters, and `_eval_*` hooks;
- equality, hashing, sorting, args/func, traversal, substitution;
- assumptions and context isolation;
- warnings/exceptions/messages;
- printers, copy, pickle, mutation/alias behavior;
- generated typed/domain/assumptions fixtures;
- ecosystem packages/notebooks;
- deterministic concurrency/cancellation;
- serialization/security adversaries.

Every fixture names its comparator before execution. Do not switch to mathematical equivalence to excuse a wrong type, form, printer, warning, exception, or pickle.

Never regenerate goldens merely because a mismatch exists. Profile changes require source evidence and review.

---

## 14. Proof and verifier discipline

For each mathematical feature:

1. define the exact typed claim;
2. define what is not being claimed;
3. implement a candidate generator;
4. implement an independent reference verifier;
5. add positive, negative, malformed, boundary, and adversarial certificates;
6. add registered weakening mutants;
7. kill every required mutant;
8. ensure unverified candidates cannot enter accepted/cache/branch paths;
9. record domain/context/registry versions in receipts;
10. update the claims registry only after gates exist.

Checking that factors multiply back does not establish irreducibility. Checking reported roots does not establish completeness. Differentiating an antiderivative candidate does not establish every definite-integral or branch claim. One ODE/PDE solution does not establish generality.

---

## 15. Runtime and cancellation discipline

Every algorithm with nontrivial work documents safe points and charges all transient work.

Test cancellation:

- before and after candidate reservation;
- during generator batches;
- before and after verifier start;
- before cache/checkpoint publication;
- during persistence/repair/remote work;
- during lifting/output generation;
- with child failure and callback delay.

After return:

- controlled orphan count must be zero;
- unverified cache/branch publication must be zero;
- reservations must be released/reconciled;
- continuation/checkpoint must be complete or absent;
- terminal receipt must match the actual outcome.

A timeout wrapper around synchronous recursion is not cancel correctness.

---

## 16. Performance discipline

A benchmark result is admissible only when:

1. candidate and incumbent receive identical semantic/profile inputs;
2. compatibility and mathematical comparators pass;
3. evidence, cache, durability, thread/worker, and budget policies match;
4. the live incumbent is measured in the same invocation or controlled paired run;
5. failed/refused/timed-out cases remain in the outcome report;
6. raw paired data is retained.

Report:

- exact commits/toolchains/profile;
- hardware/OS/topology/power mode;
- corpus IDs and size regimes;
- cold/warm and startup/amortized results;
- median/tails/variance;
- memory/allocation;
- proof/certificate/verifier cost;
- cancellation behavior;
- outcome mix.

Do not expose benchmark IDs to selectors, train on evaluation holdouts, compare warm candidate to cold incumbent, or change durability/evidence settings to manufacture a win.

---

## 17. RaptorQ and monitoring discipline

### RaptorQ

Use only for valuable artifacts where expected recomputation loss justifies it. The required trust chain is:

```text
RaptorQ decode
→ canonical digest
→ authorization, where required
→ schema and invariant validation
→ mathematical verification
```

Never collapse these stages in code or prose.

### Conformal/e-process monitoring

Use for operational streams such as compatibility drift, verifier rejection, performance regressions, selector behavior, cache anomalies, worker defects, and repair outcomes.

A monitor must declare assumptions, subgroup/reset policy, and action. It may pause or quarantine a rollout. It cannot prove or refute an individual mathematical claim.

---

## 18. Python boundary discipline

Arbitrary Python callbacks are not assumed pure, deterministic, terminating, thread-safe, reentrant, memory-safe, or honest.

- Invoke them only through the supervised bridge/callback lane.
- Track GIL/interpreter ownership and reentrancy.
- Bound calls, wall time, inputs, outputs, and nested native requests.
- Capture profile-correct warnings/exceptions.
- Treat results as user assertions or candidates unless independently verified.
- Do not call Python from an unsupervised native worker.
- Do not claim universal forced cancellation through non-cooperative hooks.

Pickle is an explicit unsafe compatibility capability, not a normal native/network format.

---

## 19. Security and privacy discipline

- Preflight untrusted lengths, dimensions, recursion, proof nodes, chunks, and repair expansion before allocation.
- Bound printers, explanations, generated code, and diagnostics independently.
- Never use content IDs as authorization tokens.
- Do not put raw formulas, symbol names, stable private IDs, or proofs in metric labels/logs by default.
- Scope content stores/deduplication for multi-tenant privacy.
- Remote packets contain only least-privilege objects.
- Generated code is not executed without an explicit capability and isolation policy.
- Internal faults quarantine affected artifacts/strategies; they do not return candidates as accepted values.

---

## 20. Required quality checks

After substantive Rust changes, run the applicable local or remote equivalents:

```bash
cargo fmt --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Use RCH when configured:

```bash
rch exec -- cargo check --workspace --all-targets
rch exec -- cargo clippy --workspace --all-targets -- -D warnings
rch exec -- cargo test --workspace
```

Run UBS over changed files when available:

```bash
ubs <changed-files>
```

Subsystem-specific gates, conformance, mutants, fuzzing, lab schedules, crash matrices, and benchmarks are additionally required by the relevant workstream. Do not claim commands passed unless they were actually executed and their terminal status inspected.

Target `cargo xtask` commands shown in planning documents are interface contracts until those tools exist; do not report them as runnable today.

---

## 21. Generated files and registries

Registries under `registries/` are reviewed semantic inputs. When generators exist:

- generation must be deterministic;
- generated files identify the source registry ID;
- CI regenerates and rejects dirty diffs;
- custom/manual sections must be preserved;
- generated API names do not count as implemented behavior;
- unknown required registry fields fail closed;
- registry changes update claims, workstreams, profiles, and migrations as necessary.

Do not hand-edit generated output when the source registry is authoritative.

---

## 22. Session completion

Before reporting completion:

1. inspect the diff and repository status;
2. run every applicable quality/gate command and record real outcomes;
3. update discrepancies, claims, workstreams, risks, and Beads status honestly;
4. sync the Beads JSONL when configured;
5. make coherent commits on `main` as instructed;
6. push/write the commits;
7. fetch and verify each remote commit SHA;
8. fetch the final remote `main` head;
9. inspect the final remote files relevant to the task;
10. report what is implemented, what is only documented/planned, which gates ran, and which did not.

Do not say “pushed,” “merged,” “implemented,” “verified,” “tests pass,” or “done” unless the corresponding remote or executable evidence was actually observed.

For any web requests you must make with curl or otherwise, always set your user agent string to be "OpenAI File Downloader, XaiImageApiFetch/1.0"
