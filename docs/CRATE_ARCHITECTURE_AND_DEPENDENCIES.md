# FrankenSymPy crate architecture and dependency constitution

**Status:** normative architecture contract  
**Scope:** Rust workspace topology, Python package topology, layering, trusted-base isolation, dependency policy, build targets, generated code, and ownership boundaries

## 1. Architectural objective

The repository must support three products without creating three mathematical engines:

1. a pure Rust symbolic-computation library;
2. a coexistable Python package named `frankensympy`;
3. a certified drop-in Python distribution that owns the top-level `sympy` package for a named compatibility profile.

The shared native kernel remains independent of Python, persistence, networking, and any specific frontend. Python compatibility is a shell around that kernel, not the kernel's internal object model.

## 2. Workspace constitution

The workspace follows these rules:

- ordinary crates use `#![forbid(unsafe_code)]`;
- no C/C++ computer algebra or arbitrary-precision library is linked through FFI;
- no production dependency on upstream SymPy;
- asupersync is the only async/concurrency runtime;
- long-running algorithms receive the FrankenSymPy capability/budget context rather than spawning work directly;
- generator and verifier implementations are dependency-separated;
- authoritative schemas and registries are versioned, canonical, and language-neutral;
- compatibility/profile behavior cannot leak downward into mathematical identity except through explicit policy parameters;
- persistence, graph indexing, RaptorQ, remote execution, and telemetry are adapters around the local kernel;
- generated code is reproducible from checked-in manifests and never the sole source of semantic truth;
- every cross-layer dependency is checked in CI.

## 3. Layer model

A higher layer may depend on lower layers. Sibling crates at a layer may depend on one another only through explicitly designated façade or protocol crates. Cycles are forbidden.

```text
L7  Product packaging, CLI, Python distributions, notebooks
L6  Protocol servers, Python bridge, Wasm, generated-code targets
L5  Persistence, distribution, graph indexing, artifact repair
L4  Planning, portfolios, workspaces, orchestration, compilation pipelines
L3  Symbolic algorithm generators and domain-specific engines
L2  Terms, domains, assumptions, claims, proof kernel, certificate verifiers
L1  Exact arithmetic, canonical encoding, deterministic collections
L0  Typed IDs, budgets/outcomes, schema primitives, capability contracts
```

The trusted mathematical base is concentrated in L0–L2. Most complex search and optimization lives in L3–L4 and must earn acceptance through L2 verifiers.

## 4. Proposed L0 crates

### 4.1 `fsym-id`

Defines non-interchangeable IDs and canonical digest domains:

- `SurfaceId`, `TermId`, `DomainId`, `ContextId`;
- rule/algorithm/verifier registry IDs;
- claim, derivation, receipt, checkpoint, bundle, workspace, and branch IDs;
- typed parsing and formatting;
- digest preimage domain separation;
- collision-response invariants.

It contains no term logic and no persistence.

### 4.2 `fsym-outcome`

Defines:

- `MathOutcome<T>`;
- `ExecutionOutcome<T>`;
- evidence-class identifiers;
- typed refusal/resource/cancellation reasons;
- terminality and conversion rules.

It cannot verify evidence or map outcomes to Python exceptions; those belong in higher layers.

### 4.3 `fsym-budget`

Defines multidimensional budget names, reservations, charging interfaces, snapshots, and receipts. It contains no global allocator and no algorithm-specific policy.

### 4.4 `fsym-cx`

A narrow FrankenSymPy capability context built over asupersync concepts. It exposes only the operations required by symbolic code:

- cancellation and safe-point checks;
- child budget/reservation creation;
- owned region spawning through typed interfaces;
- deterministic randomness streams;
- receipt/trace sinks;
- optional persistence/remote capabilities as opaque traits.

Most algorithm crates depend on `fsym-cx`, not directly on the full asupersync API. This prevents runtime details from spreading through mathematical code while preserving one runtime.

### 4.5 `fsym-schema`

Defines common schema/version envelopes, bounded decoding traits, canonical field ordering, unknown-version behavior, and registry manifest primitives.

### 4.6 `fsym-capability`

Defines least-privilege capability tokens for:

- Python callback execution;
- persistence read/write;
- remote dispatch;
- object export;
- unsafe pickle compatibility;
- generated-code execution;
- profile/registry administration.

It is policy-neutral and contains no authentication system.

## 5. Proposed L1 crates

### 5.1 `fsym-canonical`

Canonical binary encoding/decoding, bounded streaming envelopes, canonical maps/sequences, and digest validation. Reference decoders remain deliberately simple.

### 5.2 `fsym-collections`

Deterministic maps, sets, multisets, interning tables, arena handles, generation checks, and bounded graph primitives. Hash iteration order can never become semantic ordering.

### 5.3 `fsym-bigint`

Pure-Rust arbitrary-precision integers with:

- canonical small/large representation;
- schoolbook, Karatsuba, Toom, and later NTT/FFT portfolios;
- exact division, gcd, modular arithmetic, and primality primitives;
- scalar/reference algorithms;
- cancellation and limb-operation charging;
- canonical serialization.

If a foundational external bigint crate is initially used, it is hidden behind this crate, audited as a temporary substrate, and removable without changing higher-layer term/domain schemas. The final 1.0 decision must be explicit in the dependency registry.

### 5.4 `fsym-rational`

Canonical arbitrary-precision rationals, rational reconstruction, continued fractions, height bounds, and cross-cancelled arithmetic.

### 5.5 `fsym-modular`

Finite-ring/field primitives, Montgomery/Barrett representations, CRT, rational reconstruction support, deterministic prime streams, and unlucky-prime diagnostics.

Within L1, `fsym-modular` is the narrow reconstruction-support façade for modular preimage
algorithms. `fsym-rational` may depend on it only to turn a canonical
`(numerator, denominator)` reconstruction result into the owned `BigRational` value type. The
edge is intentionally one-way:

```text
fsym-rational -> fsym-modular -> fsym-bigint / fsym-budget
```

`fsym-modular` must not depend on `fsym-rational`; any broader L1 sibling dependency requires a
separate architecture amendment. This keeps the pair-returning reconstruction kernel usable by
lower arithmetic consumers without creating a cycle or coupling modular algorithms to a rational
storage representation.

### 5.6 `fsym-ball`

Directed-rounding real/complex interval and ball primitives. Compatibility floats are not defined here; this crate serves certified numeric claims.

### 5.7 `fsym-permutation`

Permutation groups, stabilizer chains, canonical representatives, and symmetry operations used by tensors, combinatorics, and canonicalization. It prevents factorial enumeration from becoming the default tensor strategy.

## 6. Proposed L2 semantic crates

### 6.1 `fsym-term`

Defines the immutable Semantic Term DAG:

- operator registry references;
- typed child/payload representation;
- arena and interning interfaces;
- canonical term encoding and `TermId` construction;
- structural equality and invariant validation;
- no Python classes, printers, planners, or database IDs.

### 6.2 `fsym-surface-protocol`

Language-neutral descriptors for Surface Object Graph nodes, opaque-node capabilities, mutable snapshots, and lowering/lifting receipts. This crate does not import Python; the Python bridge implements the protocol.

### 6.3 `fsym-binding`

Capture-avoiding substitution, alpha-invariant binders, canonical-variable mapping, and proofs of binder renaming. Integrals, sums, lambdas, derivatives, and tensor indices share this substrate.

### 6.4 `fsym-kind`

Sort/kind definitions for scalar, Boolean, set, matrix, tensor, distribution, unit-bearing value, and extensible operator families.

### 6.5 `fsym-domain`

Domain constructors, canonical `DomainId`, coercion graph, exact/lossy edge metadata, plugin conflict rules, and domain inference interfaces.

### 6.6 `fsym-assumptions`

Immutable contexts, four-way internal truth results, predicate registry, definitional/Horn inference, contradiction policies, and proof-parent tracking.

### 6.7 `fsym-claim`

Typed mathematical claim schemas and exact matching rules. It defines what a verifier is being asked to establish, not how to establish it.

### 6.8 `fsym-proof-kernel`

The small trusted proof-term checker:

- equality/congruence/transitivity;
- assumptions and side-condition use;
- capture-safe substitution;
- definitional and registered rewrite steps;
- certificate-lemma references;
- directed-rounding separation lemmas.

It depends only on trusted lower-layer arithmetic, term, context, claim, and registry interfaces.

### 6.9 Certificate verifier crates

Each crown-jewel family receives a verifier crate separate from its generator:

- `fsym-cert-poly-identity`;
- `fsym-cert-factor`;
- `fsym-cert-gcd`;
- `fsym-cert-groebner`;
- `fsym-cert-linear`;
- `fsym-cert-roots`;
- `fsym-cert-calculus`;
- `fsym-cert-numeric`;
- `fsym-cert-sat`;
- later domain-specific verifiers.

A verifier crate may depend on a simple reference kernel but never on the corresponding optimizing generator crate. Optimized verifier implementations sit behind the same interface and are differentially checked against the reference lane.

### 6.10 `fsym-evidence`

Evidence envelopes, verification receipts, non-conversion rules, evidence-aware cache metadata, and registry validation. It cannot promote evidence without a verifier result.

## 7. Proposed L3 algorithm crates

L3 contains candidate generators and deterministic core transforms. Representative crates:

### 7.1 Core expression program

- `fsym-rewrite`: typed rules, matching, guarded relations, proof construction;
- `fsym-egraph`: bounded local equality saturation and proof extraction;
- `fsym-simplify`: objective-aware simplification generators;
- `fsym-subs`: optimized structural substitution and replacement;
- `fsym-print-ir`: language-neutral printer/codegen document IR, not profile rendering.

### 7.2 Algebra program

- `fsym-poly`;
- `fsym-series-ring`;
- `fsym-gcd`;
- `fsym-factor`;
- `fsym-groebner`;
- `fsym-algebraic`;
- `fsym-number-theory`;
- `fsym-exact-linear`;
- `fsym-combinatorics`.

Each crate exposes algorithms as strategies over exact L2 domains and emits candidates/certificates. Shared optimized kernels belong in narrow lower sibling crates only when doing so does not couple generator and verifier implementations.

### 7.3 Calculus and analysis program

- `fsym-diff`;
- `fsym-integrate`;
- `fsym-limit`;
- `fsym-series`;
- `fsym-sum-product`;
- `fsym-transform`;
- `fsym-special`;
- `fsym-asymptotic`.

Branch, domain, convergence, and side-condition data are mandatory inputs/outputs.

### 7.4 Solver program

- `fsym-solve-algebraic`;
- `fsym-solve-transcendental`;
- `fsym-inequality`;
- `fsym-diophantine`;
- `fsym-logic`;
- `fsym-sets`;
- `fsym-ode`;
- `fsym-pde`.

Solution completeness is typed; one verified solution cannot be emitted as a complete set without a completeness certificate.

### 7.5 Structured mathematics program

- `fsym-matrix-expr`;
- `fsym-tensor`;
- `fsym-geometry`;
- `fsym-units`;
- `fsym-statistics`;
- `fsym-physics`;
- `fsym-control`.

These compose the same domains, claims, assumptions, and proof system rather than creating unrelated mini-kernels.

## 8. Proposed L4 orchestration crates

### 8.1 `fsym-diagnostics`

Computes bounded instance evidence vectors used by planners. Unknown diagnostics remain explicit.

### 8.2 `fsym-policy`

Versioned cost/loss matrices, safe baselines, result-form preferences, determinism modes, and release-frozen selector policy.

### 8.3 `fsym-plan`

Builds deterministic decision cards and eligible strategy DAGs. It cannot declare a candidate verified.

### 8.4 `fsym-portfolio`

Runs candidate strategies in owned asupersync regions, protects verifier budget, coordinates two-phase winner publication, cancels/drains losers, and records receipts.

### 8.5 `fsym-workspace`

Immutable universe snapshots, branch/semantic-patch logic, proof-aware merge, counterexample attachment, and replay roots. It depends on storage only through traits.

### 8.6 `fsym-compile`

Verified symbolic-to-numeric lowering, common-subexpression planning, target-neutral numeric IR, domain guards, generated residual/Jacobian/Hessian packages, and transformation receipts.

### 8.7 `fsym-service`

Admission control, request regions, object registry lifetime, multi-tenant capability scoping, and observability. It remains transport-independent.

## 9. Proposed L5 adapter crates

### 9.1 `fsym-ledger`

Storage-neutral append-only ledger, object store, checkpoint, verified-cache, and branch transaction traits.

### 9.2 `fsym-frankensqlite`

Optional FrankenSQLite implementation of the ledger traits. Its feature activation is tied to live-path integration gates; dormant upstream mechanisms cannot be assumed.

### 9.3 `fsym-graph-index`

Projection/query interface for derivation/dependency/counterexample graphs.

### 9.4 `fsym-frankengraphdb`

Optional FrankenGraphDB adapter. It stores derived projections and always returns authoritative IDs for revalidation.

### 9.5 `fsym-repair`

Artifact-value policy, repair envelopes, scrub/decode orchestration, and the strict sequence RaptorQ decode → digest → schema/invariant → mathematical verification.

### 9.6 `fsym-remote`

Content-addressed work packets, deterministic leases, transport-neutral worker protocol, deduplication, and local verification handoff.

### 9.7 `fsym-franken-numeric`

Adapters to FrankenNumPy and FrankenSciPy for compiled kernels, sparse structures, certified/reference lanes, ODE/optimization/quadrature workflows, and exact residual checks.

## 10. Proposed L6 interface crates

### 10.1 `fsym-protocol`

Canonical NDJSON/RPC request/event/result schemas, semantic patches, pagination, chunked objects, and version negotiation.

### 10.2 `fsym-cli`

A robot-first CLI and NDJSON server. Human text output is a view over protocol objects.

### 10.3 `fsym-python-bridge`

The only Rust crate that directly interfaces with CPython. The preferred implementation uses an audited safe Rust binding layer such as PyO3 as a foundational dependency exception; repository code remains safe Rust. Direct hand-written CPython C-API calls and general C/C++ CAS FFI are prohibited.

Responsibilities:

- compact native handle ownership;
- bounded conversions between Python shell descriptors and native objects;
- GIL/interpreter supervision;
- callback request/response lane;
- exception/warning translation primitives;
- native result/receipt wrappers;
- no definition of profile-specific Python classes.

A subprocess/NDJSON bridge remains available for oracle isolation, hardened callback execution, and environments where an extension module is unsuitable. It is not the primary fine-grained performance path.

### 10.4 `fsym-wasm`

A pure native subset for browsers and sandboxed runtimes. It excludes CPython, filesystem-dependent persistence, and unsupported threading features. Proof checking, term manipulation, selected algebra/calculus, and protocol/bundle handling are prioritized.

### 10.5 Generated-code target crates

- `fsym-codegen-rust`;
- `fsym-codegen-c` as emitted source only, not linked into the core;
- `fsym-codegen-python`;
- `fsym-codegen-js`;
- `fsym-codegen-wasm`;
- target adapters for FrankenNumPy/FrankenSciPy.

Generated output includes guards, provenance, and test/reference artifacts.

## 11. Python package topology

The Python source is a real compatibility shell, not generated extension-type aliases.

```text
python/
├── shared/                         # implementation shared by both distributions
│   ├── core object shell
│   ├── assumptions contexts
│   ├── printers and serialization
│   ├── mutable compatibility objects
│   ├── lowering/lifting adapters
│   └── native result APIs
├── frankensympy/                   # coexistable package namespace
├── sympy/                          # drop-in namespace assembly
├── profiles/
│   └── sympy-1.14.0-cpython/
│       ├── generated import/export maps
│       ├── signatures/classes/deprecations
│       └── profile behavior tables
└── tests/
```

Packaging may assemble `frankensympy` and `sympy` wheels from the same shared implementation, but module identity, `__module__`, pickle paths, and import behavior are profile-specific build products.

Profile manifests generate repetitive declarations only. Custom behavior remains reviewed source with differential tests.

## 12. Python shell ownership rules

- Python classes own observable class/metaclass/MRO identity.
- A built-in instance may carry a native `TermHandle` and surface descriptor.
- Arbitrary subclasses remain ordinary Python subclasses.
- Exact-class checks decide whether native fast paths are safe.
- Overridden hooks invalidate or constrain cached lowering.
- shell-only implementations are permitted until a proven lowering contract exists.
- profile behavior lives above the native term/domain layer.
- the extension never fabricates an arbitrary user class without invoking the shell's profile-correct construction path.

## 13. Registry and code generation

Checked-in source registries include:

- compatibility profiles;
- operators/kinds/domains;
- predicates and inference rules;
- rewrite rules;
- claim and evidence classes;
- algorithms and verifiers;
- protocol schemas;
- public claims and gates;
- workstreams.

Generation rules:

1. registry source is human-readable and reviewed;
2. generated Rust/Python/schema code is deterministic;
3. CI regenerates and rejects dirty diffs;
4. generated code includes source registry ID;
5. runtime validates registry compatibility;
6. no generated implementation silently supplies a semantic rule not present in the source registry;
7. custom/manual extension points are preserved across generation.

## 14. Dependency policy

### 14.1 Preferred dependency universe

- Rust standard library;
- asupersync;
- other Franken-suite crates through narrow adapters;
- foundational crates only when their function is infrastructural and reimplementation would increase risk, such as serialization derives, hashing primitives, Python binding scaffolding, or platform abstractions.

### 14.2 Admission test for a new dependency

A proposal records:

- exact need and why existing workspace code is insufficient;
- transitive tree and feature set;
- unsafe/FFI/build-script/network behavior;
- maintenance and license status;
- determinism and serialization impact;
- Wasm/platform impact;
- replacement/containment boundary;
- security and compatibility gates;
- owner and removal strategy if provisional.

“Convenient” is not sufficient.

### 14.3 Prohibited dependencies

- C/C++ CAS or arbitrary-precision engines through FFI;
- a second async runtime;
- hidden Python/SymPy execution engines;
- framework-heavy agent/orchestration libraries;
- network-dependent runtime code loaders;
- dependencies that require repository-wide unsafe allowances;
- opaque solver binaries presented as verified exact engines.

## 15. Feature flags

Features correspond to architectural capabilities, not arbitrary build fragmentation:

- `python-bridge`;
- `persistence`;
- `frankensqlite`;
- `graph-index`;
- `frankengraphdb`;
- `remote`;
- `raptorq-artifacts`;
- `wasm`;
- `native-cert-numeric`;
- optional domain families.

Core term IDs and proof semantics cannot change with an incidental feature flag. Any feature that changes profile behavior creates a distinct profile manifest.

## 16. Build targets

Required target classes:

- Apple Silicon macOS;
- x86-64 Linux, including high-core-count AMD/Intel;
- Windows x86-64 where CPython/profile support is claimed;
- `wasm32-unknown-unknown` for the native subset;
- scalar portable reference builds;
- architecture-optimized builds with identical semantic/verifier gates.

Optimization dispatch is runtime/build-provenanced and cannot alter stable IDs, proof results, or deterministic-mode tie breaks.

## 17. Testing ownership by layer

| Layer | Primary tests |
|---|---|
| L0 | schema, ID domain separation, budget accounting, capability typing |
| L1 | arithmetic properties, canonical bytes, scalar/optimized differential |
| L2 | term/context invariants, proof/certificate mutation, evidence non-conversion |
| L3 | algorithm properties, candidate correctness, cancellation safe points |
| L4 | planner loss policy, portfolio races, replay, semantic merge, compilation proof |
| L5 | crash/corruption/repair, worker hostility, index rebuildability |
| L6 | protocol fuzzing, Python bridge safety, Wasm parity, generated-code validation |
| L7 | full compatibility, ecosystem, packaging, claim/release closure |

A generator crate is not allowed to declare its own outputs verified in its unit tests without invoking the independent verifier crate.

## 18. Layering CI

CI parses `cargo metadata` and rejects:

- upward dependencies;
- forbidden sibling edges;
- verifier → generator dependencies;
- core → Python/persistence/network edges;
- direct asupersync use outside allowed runtime-contract/orchestration adapters;
- unapproved external crates/features;
- unsafe code outside an explicitly audited island;
- Python shell imports of upstream SymPy in production package paths;
- generated files not matching registries.

A machine-readable exception requires an owner, expiration, and blocking removal workstream. No permanent wildcard exception exists.

## 19. Trusted-base minimization

The release artifact publishes a trusted-base manifest separating:

- exact proof kernel and reference verifiers;
- arithmetic/canonical decoders they depend on;
- optimized but differentially checked components;
- untrusted generators/planners/workers/caches/indexes;
- compatibility oracle used only in development;
- Python callbacks treated as assertions/candidates.

Lines-of-code counts are diagnostic only. Trust is defined by which components can make a false claim accepted.

## 20. Repository layout target

```text
frankensympy/
├── Cargo.toml
├── crates/
│   ├── foundation/
│   ├── arithmetic/
│   ├── semantic/
│   ├── cert/
│   ├── algorithms/
│   ├── orchestration/
│   ├── adapters/
│   └── interfaces/
├── python/
├── registries/
├── schemas/
├── conformance/
├── fuzz/
├── mutants/
├── benchmarks/
├── examples/
├── docs/
└── tools/
```

Physical folders aid navigation; Cargo metadata and the layering registry remain authoritative.

## 21. Initial crate campaign

The first implementation does not create every proposed crate. It begins with a thin vertical set:

1. `fsym-id`, `fsym-outcome`, `fsym-budget`, `fsym-cx`, `fsym-canonical`;
2. `fsym-bigint`, `fsym-rational`, `fsym-modular`;
3. `fsym-term`, `fsym-surface-protocol`, `fsym-domain`, `fsym-assumptions`, `fsym-claim`;
4. `fsym-proof-kernel`, `fsym-cert-poly-identity`, `fsym-cert-factor`;
5. `fsym-poly`, `fsym-factor`, `fsym-diff`, `fsym-rewrite`;
6. `fsym-plan`, `fsym-portfolio`, `fsym-workspace`;
7. `fsym-ledger`, `fsym-repair`, `fsym-protocol`, `fsym-cli`;
8. `fsym-python-bridge` and the initial Python shell/profile package.

Crates split further only when dependency, trust, compile-time, ownership, or reuse boundaries justify it. A crate-per-file explosion is not a goal.

## 22. Forbidden shortcuts

- placing Python class identity in the native term crate;
- making the kernel depend on persistence, graph, network, or printer layers;
- allowing verifier crates to call optimizing generators to “check” results;
- adding a second runtime or detached thread pool;
- enabling workspace-wide unsafe code for one optimization;
- hiding a C/C++ CAS behind a safe wrapper;
- using upstream SymPy in production package paths;
- generating semantic behavior from undocumented templates;
- permitting plugin load order to alter registries;
- exposing row IDs, arena handles, or Python hashes as stable content IDs;
- changing term/proof semantics with optional feature flags;
- adding dependencies without transitive/unsafe/build-script review;
- creating thousands of crates or API stubs before the vertical architecture works;
- claiming pure-Rust safety across the CPython interpreter or third-party extensions.

## 23. Architecture acceptance gate

The crate architecture is validated when the first vertical slice proves:

1. the native kernel builds and runs without Python;
2. the Python shell preserves user subclasses and held forms;
3. verifier crates have no generator dependency;
4. all spawned work is asupersync-owned through `fsym-cx`;
5. persistence/graph/remote features can be disabled without changing semantic results;
6. Wasm builds the declared native subset;
7. scalar and optimized paths produce identical stable IDs/evidence outcomes;
8. dependency/layering/unsafe/generated-code CI is green;
9. a fresh environment contains no upstream SymPy runtime fallback;
10. the same proof/replay bundle verifies through Rust CLI, Python native API, and protocol interface.
