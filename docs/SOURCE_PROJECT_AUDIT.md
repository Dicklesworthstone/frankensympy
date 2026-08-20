# FrankenSymPy source-project audit and inheritance matrix

**Status:** normative planning input  
**Audit date:** 2026-08-19 (America/New_York)  
**Purpose:** record exactly which source revisions informed the FrankenSymPy design, what is inherited, what must be adapted, and what is explicitly rejected.

This document is not a list of inspirations. It is a provenance record for architectural decisions. Every inherited mechanism still has to earn its place in a symbolic-computation system; no feature is included merely because it exists elsewhere in the Franken suite.

## 1. Pinned revisions

| Project | Pinned revision | Role in this audit |
|---|---:|---|
| [asupersync](https://github.com/Dicklesworthstone/asupersync) | [`49abe3f830661c734e17b1eaa5d365d4039fb039`](https://github.com/Dicklesworthstone/asupersync/commit/49abe3f830661c734e17b1eaa5d365d4039fb039) | Structured concurrency, capability contexts, cancellation, deterministic lab execution, RaptorQ, adaptive decision machinery, anytime-valid monitoring |
| [FrankenSQLite](https://github.com/Dicklesworthstone/frankensqlite) | [`e6eda0826d9acba781d9fba4f61db88b3800275a`](https://github.com/Dicklesworthstone/frankensqlite/commit/e6eda0826d9acba781d9fba4f61db88b3800275a) | MVCC/versioning ideas, evidence-preserving persistence, crash recovery, claim fencing, compatibility/native separation |
| [FrankenGraphDB](https://github.com/Dicklesworthstone/frankengraphdb) | [`79570164feab9ea5337c9d193eae929b4f96775c`](https://github.com/Dicklesworthstone/frankengraphdb/commit/79570164feab9ea5337c9d193eae929b4f96775c) | Named architectural bets, source-of-truth versus derived-state discipline, provenance graphs, branch/merge concepts, executable gates, claim registry |
| [FrankenNumPy](https://github.com/Dicklesworthstone/franken_numpy) | [`d328412082de80d7eb08addd4eddcec002f001ed`](https://github.com/Dicklesworthstone/franken_numpy/commit/d328412082de80d7eb08addd4eddcec002f001ed) | Full-surface inventory, differential compatibility, divergence ledger, strict/hardened separation, proof-before-performance discipline |
| [FrankenSciPy](https://github.com/Dicklesworthstone/frankenscipy) | [`fc3bc82f2bf39d89f03ada74ff302d51daf89ba4`](https://github.com/Dicklesworthstone/frankenscipy/commit/fc3bc82f2bf39d89f03ada74ff302d51daf89ba4) | Condition-aware portfolios, asymmetric-loss decisions, numerical diagnostics, certificate-bearing algorithm selection |
| [SymPy 1.14.0](https://github.com/sympy/sympy/tree/16fa855354eb7bcabd3fe10993841e03b1382692) | [`16fa855354eb7bcabd3fe10993841e03b1382692`](https://github.com/sympy/sympy/commit/16fa855354eb7bcabd3fe10993841e03b1382692) | First immutable compatibility-profile candidate and object-model oracle |
| [SymPy development head](https://github.com/sympy/sympy) | [`81b519fabdbbc8e82db154dd271100ec7fb7ef32`](https://github.com/sympy/sympy/commit/81b519fabdbbc8e82db154dd271100ec7fb7ef32) | Non-certifying moving-head drift lane only |

A future audit may advance any pin, but it must do so explicitly and preserve this table as history. A compatibility claim is always made against a named immutable profile, never against the floating phrase “current SymPy.”

## 2. Executive conclusion

A true SymPy replacement cannot be an opaque Rust `Expr` exposed through Python bindings. SymPy's public contract includes Python behavior that is observable and extensible at run time:

- subclasses of `Basic`, `Expr`, `Atom`, `Function`, and user-defined function metaclasses;
- class-sensitive equality, hashing, sorting, and reconstruction through `func(*args)`;
- constructor hooks, classmethod `eval`, `_eval_*` methods, `_sympy_`, external converters, and constructor postprocessors;
- evaluated and deliberately unevaluated forms, including thread-local evaluation policy and cache interactions;
- mutable compatibility classes that are not themselves `Basic` instances;
- assumptions attached to classes and instances, local/global contexts, warning and exception behavior;
- printers, pickles, module paths, signatures, introspection, and exact Python class identity.

FrankenSymPy therefore needs a **dual-lane object architecture**:

1. A Python-compatible shell preserves observable SymPy object behavior.
2. Eligible regions lower into a deterministic native semantic kernel.
3. Arbitrary Python extensions remain valid as opaque-but-composable nodes.
4. Lowering and lifting are explicit, versioned, receipt-producing operations.
5. Upstream SymPy is an isolated development oracle, never a hidden production fallback in a certified drop-in build.

That conclusion is the highest-priority architectural constraint in this audit.

## 3. asupersync audit

### 3.1 Mechanisms examined

The pinned tree contains the runtime and evidence mechanisms relevant to FrankenSymPy:

- region-owned tasks and quiescent scope closure;
- explicit `Cx` capability contexts;
- request → drain → finalize cancellation;
- two-phase effects and cancellation-safe publication boundaries;
- budgets, virtual time, deterministic scheduling, trace replay, and lab execution;
- race combinators that drain losers before returning;
- RaptorQ implementation, adaptive emission logic, fuzz targets, performance gates, and rollout policies;
- conformal/e-process-style regression and spectral-health monitors.

### 3.2 Adopt

- **Region ownership for every spawned symbolic job.** A factorization race, Gröbner portfolio, proof check, or numeric enclosure task must have a parent scope and a drain owner.
- **Explicit budgets.** CPU steps, wall time, allocation, expression growth, recursion depth, proof size, modular primes, and remote work all consume typed nested budgets.
- **Cancellation as protocol.** A cancelled transformation must stop at a declared consistency boundary, preserve already-published evidence, and leave no orphan verifier or cache writer.
- **Deterministic lab execution.** Schedule exploration and replay are mandatory for concurrent memoization, speculative portfolios, checkpoint publication, and distributed merges.
- **Two-phase publication.** Candidate results reserve publication rights; only independently verified candidates can commit to shared caches or user-visible winner channels.
- **RaptorQ for valuable byte artifacts.** Checkpoints, proof archives, replay bundles, fuzz corpora, and distributed work packets are repair-protected where the value justifies it.

### 3.3 Adapt

- Runtime `Outcome` is extended into a mathematical result envelope that distinguishes proved, certificate-verified, numerically certified, heuristic, inconclusive, refused, cancelled, timed out, and resource-exhausted outcomes.
- Generic task budgets become symbolic budgets with algebra-aware dimensions such as term count, coefficient height, monomial count, algebraic degree, branch count, and proof-node count.
- Adaptive policies may choose which algorithm to try, but mathematical acceptance is delegated to a separate verifier. Completion order affects latency, never truth.
- RaptorQ policy is based on expected artifact loss, repair cost, and value; it is not sprayed across in-memory expression nodes or tiny disposable caches.

### 3.4 Reject

- Any claim that cancellation has a universal time bound for arbitrary Python callbacks, foreign code, or non-cooperative work.
- Detached “background” simplification, cache warming, or proof search.
- Treating an e-process, conformal score, scheduler confidence, or algorithm-selection posterior as a mathematical proof.
- Treating successful RaptorQ decoding as content authenticity, semantic correctness, or proof validity.

## 4. FrankenSQLite audit

### 4.1 Mechanisms examined

The pinned repository demonstrates:

- explicit compatibility versus native-mode boundaries;
- page/version identities, MVCC, snapshot reasoning, conflict witnesses, and replayable history;
- a durable ledger and crash-recovery mindset;
- conformance against an external oracle;
- strong claim-fencing in the README: implemented, dormant, partial, and target-state mechanisms are separated;
- RaptorQ/ECS concepts for durable artifacts while avoiding unsupported blanket claims about the live path.

### 4.2 Adopt

- **Optional computation ledger.** Long-running or expensive symbolic jobs may persist requests, profile versions, term IDs, decisions, checkpoints, certificates, receipts, and terminal outcomes.
- **Snapshot isolation for workspace state.** A derivation reads a frozen assumptions/rule/profile snapshot even while another agent extends a branch.
- **Versioned immutable records.** Rules, assumptions, profiles, proofs, and result bundles are content-addressed and append-only once published.
- **Crash recovery and forensic replay.** On restart, completed records remain complete, reserved-but-uncommitted publications are abandoned, and resumable continuations are restored only after digest and schema verification.
- **Reality-based documentation.** Target architecture, implemented runtime, and certified compatibility status must never be collapsed into one tense or one badge.

### 4.3 Adapt

- Database MVCC becomes derivation/workspace MVCC: versioned rule registries, assumptions contexts, semantic branches, proof graphs, and cache generations.
- Commit conflict becomes semantic merge conflict. Two derivation branches can merge only when their rule/profile bases are compatible and every imported proof edge verifies.
- The persistent store is optional and outside the algebraic hot path. In-memory term interning and local rewrite scheduling cannot require a database transaction.

### 4.4 Reject

- Making persistence part of expression equality or the trusted proof kernel.
- A mutable database row as the canonical identity of a mathematical term.
- Silent recovery that changes a result, assumption set, rule ordering, or compatibility profile.
- Marketing a designed durability mechanism as live before crash injection and end-to-end recovery gates exist.

## 5. FrankenGraphDB audit

### 5.1 Mechanisms examined

The pinned repository contributes a planning and governance pattern as much as a graph engine:

- a small set of named, compositional architectural bets;
- constitutional prohibitions against shortcuts that would fake progress;
- a single authoritative state substrate with rebuildable derived indexes;
- deterministic plans, certificates, decision cards, and replay;
- branch-per-agent workflows and semantic merge;
- strict crate layering, closed dependency posture, and machine-checkable milestone gates;
- a claims registry/linter that ties prose claims to evidence.

### 5.2 Adopt

- **Named bets.** FrankenSymPy's design is organized around a small number of load-bearing compositions, not an undifferentiated feature catalog.
- **Authoritative versus derived state.** Surface objects, semantic terms, assumptions snapshots, and proof edges are authoritative. Search indexes, memo tables, cost models, statistics, and graph projections are disposable and rebuildable.
- **Branch-per-agent derivations.** Agents can fork a derivation workspace, propose transformations, attach certificates, and merge only verifier-accepted edges.
- **Claims registry.** Every public compatibility, safety, proof, determinism, and performance claim names an evidence class and a gate.
- **Forbidden-shortcut constitution.** Hidden fallbacks, unverified speculative winners, string-as-IR, benchmark cherry-picking, and proof-class inflation are release-blocking violations.

### 5.3 Adapt

- FrankenGraphDB may index enormous derivation, dependency, counterexample, and collaborative-work graphs, but it is never authoritative for term equality or proof validity.
- Graph branch merge becomes proof-aware semantic merge, not last-writer-wins and not textual patch merge.
- Query-plan certificates inspire transformation-plan receipts, but a plan receipt records what was attempted; it does not by itself prove the result.

### 5.4 Reject

- Requiring a graph database for ordinary local symbolic computation.
- Letting graph reachability stand in for logical entailment.
- Treating provenance volume as evidence strength.
- Coupling core expression identity to a server, network, or mutable index.

## 6. FrankenNumPy audit

### 6.1 Mechanisms examined

The pinned repository demonstrates:

- complete public-surface inventory with structural CI locks;
- a differential oracle and machine-readable parity reports;
- strict/hardened operating modes and explicit compatibility debt;
- evidence and divergence ledgers;
- profile-before-optimize discipline;
- format hardening, fuzzing, adversarial fixtures, and repair-protected evidence bundles.

### 6.2 Adopt

- **Full surface inventory.** Module paths, exports, call signatures, classes, methods, properties, warnings, exceptions, printers, pickles, and optional-feature surfaces all become profile artifacts.
- **Discrepancy ledger.** Every mismatch is classified, minimized, owned, and tied to a closure gate. “Known difference” is not a euphemism for accepted drop-in behavior.
- **Parity before performance.** A benchmark case cannot enter an aggregate unless its semantic comparison gate passes first.
- **Strict and native/hardened separation.** Compatibility behavior is immutable within a certified profile; advanced native controls live in an explicit namespace and result envelope.
- **Machine-readable conformance artifacts.** A release claim can be recomputed from fixtures and manifests rather than trusted from prose.

### 6.3 Adapt

- Array alias/stride inventories become symbolic object-model and evaluation-policy inventories.
- Numeric tolerances become a richer comparator taxonomy: exact structural identity, mathematical equivalence under assumptions, printer equality, exception equality, set equality, certified enclosure, and explicitly ordered nondeterminism.
- Upstream fallback used during early development must be isolated, visible, and prohibited from certified drop-in wheels.

### 6.4 Reject

- Reachability of an API name as evidence that its behavior is implemented.
- Identity-preserving fallback to upstream SymPy in a release advertised as an independent drop-in replacement.
- Aggregate performance comparisons that mix compatible and incompatible cases.
- Regenerating goldens merely to make a drift gate green.

## 7. FrankenSciPy audit

### 7.1 Mechanisms examined

The pinned repository's central contribution is a condition-aware solver portfolio:

- instance diagnostics and structural evidence;
- explicit state models and asymmetric loss matrices;
- calibrated selection and fallback;
- per-call decision/audit records;
- stability-first acceptance and differential conformance.

### 7.2 Adopt

- **Proof-carrying algorithm portfolios.** Polynomial GCD, factorization, Gröbner bases, integration, limits, equation solving, exact linear algebra, and certified numerics may race multiple strategies.
- **Asymmetric loss.** A slow exact answer, a fast candidate requiring verification, an inconclusive result, and a mathematically false answer have radically different costs. The selector's objective must encode that asymmetry.
- **Instance diagnostics.** Domain, sparsity, degree, coefficient height, symmetry, branch structure, assumptions, expected proof cost, and prior verified outcomes inform planning.
- **Decision cards.** Every consequential adaptive route records evidence, alternatives, expected loss, policy version, and fallback conditions.

### 7.3 Adapt

- CASP-style confidence selects work; it never upgrades evidence. A candidate becomes accepted only through a verifier appropriate to its claim.
- Numerical backward-error and condition diagnostics become one evidence family among many, alongside exact certificates and kernel-checked rewrites.
- Portfolio learning occurs across profile-scoped telemetry and is guarded against workload drift, subgroup regressions, and reward hacking.

### 7.4 Reject

- A posterior probability or low expected loss as proof that an identity is true.
- A heuristic simplifier returning an ordinary expression indistinguishable from a proved transformation in native mode.
- Self-reported speedups without a live incumbent in the same invocation and semantic parity on the measured case.

## 8. SymPy 1.14.0 object-model audit

The first compatibility candidate is pinned to SymPy 1.14.0 rather than a floating development branch. The following observations are architectural requirements.

### 8.1 `Basic` is a Python protocol, not merely a node layout

At the pinned revision, `Basic`:

- prepares class assumptions from `__init_subclass__`;
- stores `_args`, `_mhash`, and assumptions state;
- reconstructs through `self.func(*self.args)`;
- defines class-sensitive structural hashing and equality;
- participates in external conversion through `_sympy_` and converter registries;
- exposes canonical class ordering and recursively computed sort keys;
- supplies pickle hooks and traversal/replacement behavior inherited and overridden throughout the library.

A native term handle can accelerate a built-in object, but it cannot replace this protocol for arbitrary Python subclasses.

### 8.2 Functions are metaclass-extensible

`FunctionClass` is a metaclass. Undefined functions are dynamically created classes. Subclasses define classmethod `eval`, arity, `_eval_*` hooks, assumptions handlers, and custom evaluation behavior. `Function('f')` returns a class; applying it returns an instance of an undefined-function subclass. Exact signatures and exception text can be ecosystem-visible.

The compatibility shell must preserve this behavior. Native lowering treats unknown function classes as opaque semantic operators unless they explicitly declare a safe, versioned lowering contract.

### 8.3 Evaluation policy is observable state

`Add`, `Mul`, functions, and other constructors consult explicit `evaluate=` arguments or thread-local global parameters. Unevaluated construction preserves argument multiplicity and order that canonical semantic terms may intentionally erase. Cache state interacts with evaluation-policy changes.

Therefore surface form and semantic canonical form cannot be the same graph.

### 8.4 Mutable compatibility objects exist

Not all SymPy-compatible values are immutable `Basic` instances. Mutable matrices and related classes have distinct identity, mutation, conversion, and pickling behavior. A complete drop-in profile must inventory them rather than assuming every public value is a hash-consed term.

### 8.5 Assumptions are not a Boolean annotation bag

Assumptions include tri-valued answers, inference, class and instance state, old and new APIs, contextual queries, contradictions, unknowns, and behavior implemented by user hooks. Native reasoning must preserve `True`/`False`/`None`, context identity, and proof provenance; it must never coerce “not proved” into false.

## 9. Cross-project inheritance matrix

| Mechanism | Adopt | Adapt | Reject / fence |
|---|---|---|---|
| Structured concurrency | All spawned work region-owned | Symbolic budget dimensions and verifier-owned publication | Detached tasks and universal cancellation claims |
| RaptorQ | Valuable artifacts, checkpoints, bundles | Expected-loss-driven redundancy | Hot-path term storage; authenticity or truth claims |
| Conformal e-processes | Drift/regression/anomaly monitoring | Profile/subgroup-aware telemetry | Mathematical proof or per-result correctness |
| MVCC/history | Workspace/rule/profile snapshots | Proof-aware branch merge | Persistent row IDs as term identity |
| Graph indexing | Optional giant provenance/dependency indexes | Rebuildable projection over authoritative records | Graph reachability as proof |
| Differential oracle | Development and release conformance | Isolated-process immutable profiles | Hidden production fallback |
| Solver portfolios | Race diverse strategies | Acceptance only after independent verification | Winner-by-completion-order |
| Evidence ledger | Decisions, transformations, claims, recoveries | Typed mathematical evidence lattice | Unstructured logs as certification |
| Strict/hardened modes | Explicit compatibility/native postures | Profile-bound packaging and namespaces | Silent behavior-changing hardening |
| Claims registry | Public claim → gate → artifact | Mathematical evidence-specific classes | Aspirational badges |

## 10. Dependency posture

The intended core dependency universe is deliberately narrow:

- Rust `core`/`alloc`/`std` and the pinned toolchain;
- `asupersync` for structured concurrency, deterministic lab execution, transport, and RaptorQ;
- owned Franken-suite components when their contract is narrow and their use is optional or layered;
- a very small set of foundational crates only when reimplementation would add risk without strategic value.

FrankenSymPy must not depend on upstream SymPy at production runtime. Python is necessarily present for the Python compatibility shell; the independent native Rust API must remain usable without embedding Python. C/C++ CAS libraries and opaque FFI engines are prohibited.

## 11. Audit-derived design obligations

The comprehensive plan must satisfy all of the following:

1. Define the Python shell/native kernel boundary precisely.
2. Define separate surface, semantic, and derivation/provenance representations.
3. Define stable content identity separately from Python's process-local hash behavior.
4. Define immutable compatibility profiles and a public discrepancy ledger.
5. Define a small trusted verifier kernel and evidence classes that cannot be inflated by selectors or monitors.
6. Define typed budgets, cancellation boundaries, replay, and resumable continuations.
7. Define where persistence, graph indexing, RaptorQ, and e-processes are useful and where they are forbidden.
8. Define complete conformance beyond the upstream test suite, especially Python subclassing and deliberately unevaluated forms.
9. Define parity-gated benchmarks and mutation-tested verifiers.
10. Define an end-to-end first slice that proves the architecture before surface-area expansion.
11. Define machine-verifiable milestone gates and forbidden shortcuts.
12. Keep every present-tense implementation claim tied to evidence from the repository's actual current state.

## 12. Re-audit protocol

A source pin may be advanced only through a commit that:

1. records old and new SHAs;
2. summarizes material source changes;
3. identifies affected FrankenSymPy contracts;
4. updates adopt/adapt/reject decisions where necessary;
5. updates compatibility-profile manifests and discrepancy expectations;
6. runs the relevant document/registry consistency gates.

This file is the root of the architectural provenance chain. The comprehensive plan may synthesize and extend it, but may not silently contradict it.