# FrankenSymPy conformance laboratory and benchmark program

**Status:** normative architecture contract  
**Scope:** immutable oracle profiles, inventory generation, differential/metamorphic testing, ecosystem corpora, verifier/runtime fault testing, benchmark admission and reporting

## 1. Principle

A drop-in replacement is an empirical theorem about observable behavior. A high-performance computer algebra system is a collection of mathematical theorems, numerical guarantees, and operational guarantees. FrankenSymPy tests those claims separately and composes them only when all relevant gates pass.

No benchmark result is admitted unless the benchmark case first passes its declared semantic/compatibility comparator in the same run.

## 2. The conformance laboratory

The laboratory consists of isolated runners:

```text
fixture generator
├── upstream SymPy oracle process
├── FrankenSymPy compatibility process
├── FrankenSymPy native/reference process
├── mathematical verifier process/lane
├── comparator and minimizer
├── discrepancy ledger writer
└── artifact/benchmark publisher
```

Processes do not share Python objects, caches, class registries, or imports. The harness captures exact environment fingerprints.

## 3. Immutable oracle profile

For every profile the oracle environment fixes:

- upstream repository and commit;
- source and test tree digests;
- Python implementation/version/ABI;
- OS/architecture/platform tag;
- dependency versions and optional-feature set;
- environment variables, locale, timezone, hash seed;
- cache/evaluation/printing settings;
- test inventory and exclusions;
- comparator registry.

The first target is SymPy 1.14.0 at `16fa855354eb7bcabd3fe10993841e03b1382692`. A moving-head lane detects future drift but cannot certify a release.

## 4. Public-surface inventory

A source/reflection inventory records:

- packages/modules/submodules and import aliases;
- `__all__` and top-level exports;
- classes, metaclasses, MROs, bases, slots, descriptors;
- functions/methods/properties and signatures;
- positional-only/keyword-only/variadic parameters;
- defaults, annotations where observable, deprecations;
- constants/singletons and identity relations;
- exceptions/warnings and inheritance;
- printer classes/settings;
- serialization hooks;
- optional dependency gates;
- documented and ecosystem-used private surfaces.

The inventory is structural CI input. Missing names, changed signatures, or class hierarchy deltas become discrepancies even before behavioral fixtures run.

## 5. Behavioral probe inventory

Automated probes inspect:

- construction with typical, boundary, invalid, and custom objects;
- `evaluate=True`, `False`, omitted, and context-manager behavior;
- type, class identity, MRO, `isinstance`, `issubclass`;
- `args`, `func`, `_hashable_content`, copy, deepcopy, reconstruction;
- equality, hash under multiple seeds, sort keys, comparison;
- free/bound symbols, traversal, substitution, replacement;
- assumptions queries and context nesting;
- warnings/exceptions/messages/stack policy;
- `str`, `repr`, pretty ASCII/Unicode, LaTeX, code printers;
- pickle protocols and cross-process round trips;
- mutation/alias behavior for mutable objects;
- introspection and dynamic `Function` class creation;
- external converters, `_sympy_`, constructor postprocessors;
- custom `_eval_*` hooks and overridden methods;
- optional-feature presence/absence.

Probe generation is profile-versioned and its coverage is itself auditable.

## 6. Upstream test suite

The complete applicable upstream suite is imported as an oracle asset, with:

- original test path and source digest;
- environment/marker requirements;
- port status;
- result under upstream and FrankenSymPy;
- exclusion reason and owner, if any;
- related discrepancy IDs.

Tests are not edited merely to match FrankenSymPy. A legitimate adaptation for harness isolation preserves the original and records the transformation.

Passing upstream tests is necessary, not sufficient: tests often under-specify class identity, generated edge cases, concurrency, serialization, and ecosystem interactions.

## 7. Differential fixture generation

### 7.1 Typed expression grammars

Generators are aware of:

- class/object category;
- domain and coefficient characteristic;
- commutativity;
- assumptions contexts, including unknown and contradictory states;
- evaluated/held construction;
- binders and alpha-renaming;
- branch cuts and singularities;
- matrix/tensor dimensions;
- mutable/immutable boundaries;
- custom Python subclasses and opaque nodes;
- precision and numeric special values;
- expected resource amplification.

They produce valid and deliberately invalid calls.

### 7.2 Shape control

Fixture size is controlled by semantic dimensions rather than raw string length:

- DAG nodes and sharing;
- depth;
- polynomial degree/term count/coefficient height;
- matrix size/density/structure;
- branch/piecewise count;
- binder nesting;
- proof/solution-set expected size.

### 7.3 Observations

Each side emits a normalized observation envelope:

```text
Observation
├── fixture/profile/environment IDs
├── return/exception/timeout class
├── type/class/module/MRO descriptors
├── structural/surface representation
├── printer outputs
├── warning trace
├── assumptions observations
├── mutation/alias observations
├── pickle/copy observations
├── mathematical normalization or enclosure, when requested
├── resource outcome
└── raw artifact references
```

The comparator cannot infer omitted fields.

## 8. Comparator discipline

Every fixture names a comparator from the immutable profile registry. Comparator selection is based on the API contract, not on which choice makes the test pass.

Examples:

- constructors and printers generally require exact surface/type observations;
- exact algebra routines may require profile type/form plus independently verified mathematical claims;
- sets require declared exact/membership/completeness semantics;
- numeric routines use profile-specific tolerance, NaN, signed-zero, and branch policies;
- genuinely nondeterministic APIs use frozen envelopes only after upstream nondeterminism is demonstrated.

Comparator changes create a profile diff and trigger re-review of every newly admitted result.

## 9. Custom subclass corpus

This is a release-critical suite, not a niche extension test.

Corpus classes vary:

- direct `Basic`, `Atom`, `Expr`, `Function`, and matrix subclasses;
- custom `__new__`, `__init_subclass__`, slots, attributes, equality/hash;
- classmethod `eval` and `nargs` behavior;
- `_eval_is_*`, `_eval_derivative`, `_eval_rewrite`, `_eval_subs`, `_eval_evalf`;
- custom printer methods;
- dynamic/local classes and pickle limitations;
- external converter and `_sympy_` objects;
- noncommutative and unusual kind/domain declarations;
- callbacks that raise, recurse, mutate global state, delay, or return invalid objects.

Tests mix custom nodes deeply with native-backed built-ins and ensure shell operations never erase class identity or assume unsafe semantics.

## 10. Held-form corpus

Fixtures cover:

- `Add`, `Mul`, `Pow`, functions, relations, derivatives, integrals, sums, matrices, and containers with evaluation disabled;
- nested and mixed held/evaluated forms;
- duplicate identities and noncanonical argument order;
- printer, traversal, `func(*args)`, copy/pickle, substitution behavior;
- lowering to shared semantic terms and lifting back;
- evaluation context changes and cache clearing;
- concurrent/thread/task-local policy behavior.

Mathematical equivalence cannot excuse surface drift in this corpus.

## 11. Metamorphic testing

Metamorphic relations supplement differential tests.

### 11.1 Structural metamorphisms

- pickle/copy/reconstruct round trip preserves declared observations;
- lower → lift preserves profile surface where promised;
- serialize → parse preserves stable IDs for canonical native objects;
- deterministic replay preserves terminal digest;
- mutable → immutable → mutable conversion follows profile semantics.

### 11.2 Mathematical metamorphisms

Under explicit domains/assumptions:

- substitution preserves proved equality;
- derivative linearity/product/chain rules;
- factor product and multiplicity;
- GCD divisibility/Bézout properties;
- Gröbner ideal membership and S-polynomial reductions;
- solve substitution plus completeness obligations;
- matrix decomposition identities;
- interval enclosure monotonicity and refinement;
- units/dimensions invariants;
- code-generated evaluator agrees with exact/reference lane.

A metamorphic property is not used when its side conditions are unknown.

## 12. Mathematical oracle independence

Upstream SymPy is a compatibility oracle, not the only mathematical oracle.

Independent checks include:

- native scalar/reference algorithms;
- proof/certificate verifiers;
- exact brute force for small domains;
- independently generated known-answer corpora;
- cross-system artifacts used only as candidate/reference data and never hidden production fallback;
- formal identities and published benchmark sets with provenance;
- certified numerical enclosures.

When upstream and FrankenSymPy agree but an independent verifier rejects the claim, the case is a mathematical incident, not a pass.

## 13. Fuzzing

Fuzz targets include:

- term/surface/bundle/proof/certificate decoders;
- Python-shell constructors and converters;
- assumptions contexts and rule matching;
- arithmetic/domain/coercion kernels;
- printers/parsers/serialization;
- rewriting/e-graph boundaries;
- every certificate verifier;
- NDJSON/RPC and semantic patches;
- persistent cache/checkpoints/RaptorQ envelopes;
- remote work packets/responses.

Fuzzing uses structure-aware generators and retains minimized counterexample bundles with exact environment/universe IDs.

## 14. Mutation testing

Mutation suites target:

- rewrite side conditions and direction;
- assumptions implications;
- canonical term encoding;
- equality/hash/sort behavior;
- lowering/lifting class checks;
- certificate completeness criteria;
- directed rounding;
- cache-key universe fields;
- cancellation/two-phase publication;
- comparator strictness;
- claims registry/gates.

Registered mutants must be killed. Surviving mutants block the related claim or evidence class.

## 15. Concurrency and deterministic testing

The lab explores:

- interning races;
- cache read/write/invalidation;
- profile/context-local state;
- speculative winner/verification races;
- cancellation at every safe point;
- checkpoint/GC/repair races;
- branch merge conflicts;
- remote duplicate/late/byzantine responses;
- Python hook reentrancy/delay/exception;
- varied core counts and scheduling.

Assertions include zero controlled orphans, no unverified cache publication, stable semantic outcomes, and replayable traces.

## 16. Persistence and corruption testing

- crash after each publication step;
- torn/truncated/missing objects;
- wrong schema/context/registry dependencies;
- stale branch/cache generations;
- RaptorQ loss/corruption matrices;
- repaired bytes failing/satisfying canonical digest;
- proof re-verification after recovery;
- graph index deletion/rebuild;
- checkpoint resume across fresh processes;
- malicious stored `verified` flags.

Byte recovery and mathematical verification are asserted separately.

## 17. Ecosystem conformance

A versioned corpus includes:

- packages importing public and widely used semi-private SymPy surfaces;
- notebooks and scripts from representative scientific/engineering domains;
- generated code workflows;
- pickled artifacts where redistribution permits;
- documentation examples and tutorials;
- interactions with NumPy/SciPy/pandas/plotting/Jupyter/tooling;
- custom-function libraries;
- agent-generated workloads.

Each corpus entry records licensing/provenance, exact dependency lock, expected outputs, nondeterminism policy, runtime budget, and owner.

Ecosystem success is reported by named corpus/version, never as “works with the ecosystem” without scope.

## 18. Differential minimization

When a mismatch appears, the minimizer preserves the relevant observation and reduces:

- expression structure and assumptions;
- custom class implementation;
- environment/optional dependencies;
- request sequence;
- concurrency schedule;
- persistence artifacts;
- proof/certificate payload.

Minimization is semantics-aware: it does not canonicalize away the held form, class identity, branch, or context that causes the failure.

The minimized reproducer becomes a permanent regression fixture and discrepancy attachment.

## 19. Discrepancy workflow

1. capture immutable failure bundle;
2. minimize without altering comparator;
3. classify severity/surface/root-cause hypothesis;
4. determine whether upstream behavior is intended, accidental, or a suspected upstream bug;
5. add closure tests before or with the fix;
6. update profile inventory if the oracle contract itself legitimately changes;
7. close only when the original and minimized fixtures pass and no broader mutation survives.

A mismatch is never erased by changing a golden without this workflow.

## 20. Benchmark suites

### 20.1 Compatibility microbenchmarks

- construction, hashing, sorting, substitution, assumptions, printers;
- built-in and custom subclass mixtures;
- held versus evaluated forms;
- shell/kernel boundary costs;
- pickle/serialization.

### 20.2 Algebra kernels

- big integer/rational arithmetic;
- dense/sparse polynomial arithmetic;
- GCD/factorization/Gröbner;
- exact dense/sparse linear algebra;
- root isolation/algebraic numbers;
- Boolean/SAT and set operations.

### 20.3 Calculus and solvers

- differentiation/Jacobian/Hessian;
- integration families;
- limits/series/asymptotics;
- equation/inequality/Diophantine;
- ODE/sum/transform workloads.

### 20.4 End-to-end and agent workloads

- model residual/Jacobian generation and numeric compilation;
- theorem/proof exploration with branches;
- cancellation/checkpoint/resume;
- distributed portfolio and local verification;
- large proof graph queries;
- notebook/ecosystem flows.

### 20.5 Adversarial workload suite

- expression swell bombs;
- high coefficient height;
- unlucky modular primes;
- pathological Gröbner pair growth;
- branch-heavy piecewise forms;
- deeply nested binders;
- malicious custom hooks/serializations;
- cache/repair corruption.

## 21. Benchmark admission pipeline

For each case:

1. construct identical canonical/surface input bundle;
2. run upstream/live incumbent and FrankenSymPy candidate in the same invocation or tightly controlled paired environment;
3. collect semantic/compatibility observations;
4. run required mathematical verifier;
5. exclude and ledger any failing case;
6. only then record timed samples and resource metrics;
7. publish paired raw data and aggregate summary.

A candidate can be faster on a wrong answer, but that result appears only in the failure ledger, never the speed aggregate.

## 22. Measurement contract

Reports include:

- exact commits/profile/build flags/toolchains;
- hardware/OS/CPU topology and power policy;
- input corpus IDs;
- cold/warm cache and persistence/durability modes;
- thread/core/worker counts;
- evidence requirement and verifier cost;
- median, p90, p95, p99, variance/confidence interval;
- peak/live memory and allocations where available;
- proof/certificate size;
- outcome mix including timeout/refusal/inconclusive;
- amortization and startup separated;
- live incumbent results.

Benchmarks do not hard-code fixture IDs into algorithm selectors. Hidden and generated holdouts detect this.

## 23. Performance acceptance

A change is a performance win only if:

1. relevant semantic and compatibility gates remain green;
2. the live incumbent is measured in the same invocation;
3. the primary metric improves in the declared workload class;
4. no protected p99, memory, proof-size, cancellation, or neighboring-workload budget regresses beyond policy;
5. the optimization does not weaken a verifier, comparator, profile, durability mode, or evidence requirement;
6. raw evidence is published.

A self-comparison to an older FrankenSymPy build without the incumbent is maintenance evidence, not a leapfrog claim.

## 24. Performance targets

Targets are workload-relative rather than invented fixed milliseconds before implementation. Each milestone establishes:

- a correctness-complete corpus;
- upstream SymPy baseline;
- current FrankenSymPy reference lane;
- target speedup/memory envelope by size regime;
- crossover point where native lowering pays off;
- tail and cancellation budgets;
- maximum shell-compatibility overhead;
- proof-generation/verification overhead.

Early goals prioritize large exact workloads and batch/agent flows where Rust representations, parallel portfolios, and compilation can create orders-of-magnitude opportunity. Small scalar calls must avoid disastrous boundary overhead but need not win every microbenchmark immediately.

## 25. Release artifacts

A release candidate publishes:

- compatibility profile and inventory manifests;
- upstream and generated conformance reports;
- open/closed discrepancy ledger digest;
- verifier mutation report;
- fuzz/adversarial corpus digest;
- concurrency/replay report;
- persistence/repair report for shipped features;
- parity-gated benchmark raw data and summary;
- claim registry resolution;
- build provenance/SBOM/dependency audit;
- optional RaptorQ sidecars for valuable evidence packs.

All artifacts point to the same release commit.

## 26. CI topology

Suggested blocking lanes:

| Gate | Scope |
|---|---|
| G1 | formatting, lint, dependency/layering, schema validation |
| G2 | unit and property tests |
| G3 | object-model and compatibility inventory |
| G4 | upstream differential suite |
| G5 | generated differential and metamorphic suites |
| G6 | proof/certificate verifier and mutation suites |
| G7 | fuzz/adversarial/security corpus |
| G8 | deterministic concurrency/cancellation/replay |
| G9 | persistence/crash/repair if affected/shipped |
| G10 | ecosystem corpus |
| G11 | parity-gated performance regressions |
| G12 | claims/discrepancy/release artifact closure |

Change-aware CI can select subsets, but release certification reruns the full required matrix on one commit.

## 27. Anti-reward-hacking rules

Release-blocking violations include:

- weakening tests/comparators/gates to land code;
- deleting hard fixtures or excluding them without ledger review;
- regenerating goldens after a mismatch without profile-change evidence;
- measuring only successful cases;
- benchmark-path hard-coding;
- claiming independent verification from shared faulty code;
- counting API stubs/reachable names as parity;
- treating upstream test count as total coverage;
- reporting proof-node count as proof quality;
- hiding persistence/verifier overhead;
- comparing warm candidate to cold incumbent;
- changing durability/evidence/profile settings between competitors;
- training selector policy on the benchmark test set.

## 28. Initial conformance campaign

Before broad API expansion, the first slice must achieve:

1. complete inventory for the initial core object subset;
2. upstream differential fixtures for construction/equality/hash/sort/evaluate/assumptions/printers/pickles;
3. custom subclass and opaque-node corpus;
4. generated held/evaluated expression grammar;
5. native proof/certificate checks for differentiation and polynomial factorization;
6. cancellation/schedule exploration for the first portfolio;
7. checkpoint corruption/repair/resume drill;
8. parity-gated benchmarks against upstream SymPy and a scalar native reference;
9. minimized discrepancy bundles and machine-readable ledger;
10. a release-style artifact bundle showing every claim remains `planned` until its gate exists.
