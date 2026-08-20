# FrankenSymPy dependency-aware workstream graph

**Status:** normative implementation program  
**Scope:** architectural workstreams, dependencies, acceptance gates, parallelism, critical path, Beads conversion, milestone closure

## 1. Purpose

This document turns the architecture into an executable dependency graph. A workstream is not “done” because code exists or an agent reports completion. It closes only when its objective artifacts and named acceptance gates pass, its discrepancy/claim effects are recorded, and no forbidden shortcut was used.

The machine-readable companion is `registries/workstreams.toml`. That registry is structural truth for IDs and edges; this document supplies rationale and full gate semantics.

## 2. Graph rules

- Every workstream has one stable ID and bounded objective.
- Dependencies form a directed acyclic graph.
- A workstream may begin discovery/prototyping early, but cannot close before dependencies close.
- Generator and verifier work are separately owned even when grouped under one program.
- Cross-cutting conformance begins before broad implementation and grows with every surface.
- Claims remain `planned` until the exact gate artifacts exist on the same commit.
- Beads conversion requires objective commands, fixtures, evidence artifacts, and forbidden shortcuts.
- Structural graph changes are single-writer and registry-reviewed.

## 3. Milestone map

```text
M0  Planning substrate and claim discipline
M1  Native semantic nucleus
M2  Python object-model vertical slice
M3  First proof-carrying algebra portfolio
M4  Certified Jacobian hero pipeline
M5  Durable/agent-native execution fabric
M6  SymPy 1.14 compatibility expansion
M7  Breadth, optimization, and ecosystem closure
M8  Certified drop-in 1.0
```

Milestones are gates, not dates.

## 4. Top-level dependency graph

```text
WS00 Governance, registries, claims
├── WS01 Conformance laboratory foundation
├── WS02 Foundation types, schemas, budgets, Cx
│   ├── WS03 Exact arithmetic substrate
│   │   ├── WS04 Terms, domains, assumptions, bindings
│   │   │   ├── WS05 Python compatibility shell
│   │   │   ├── WS06 Proof kernel and evidence
│   │   │   │   ├── WS07 Verified rewriting and simplification
│   │   │   │   ├── WS08 Polynomial representations and arithmetic
│   │   │   │   │   ├── WS09 GCD/factorization and certificates
│   │   │   │   │   ├── WS10 Exact linear algebra
│   │   │   │   │   └── WS17 Gröbner/ideal algebra
│   │   │   │   ├── WS11 Certified numerics and algebraic numbers
│   │   │   │   └── WS12 Differentiation and symbolic compilation
│   │   │   └── WS13 Structured portfolios and runtime replay
│   │   └── WS14 Agent protocol and semantic workspaces
│   └── WS15 Persistence, checkpoints, and RaptorQ repair
│       └── WS16 Remote workers and graph indexing
├── WS18 Integration, limits, series, transforms
├── WS19 Solvers, sets, logic, ODE/PDE
├── WS20 Structured mathematics domains
├── WS21 Compatibility and ecosystem closure
├── WS22 Performance and architecture optimization
└── WS23 Packaging, release, and 1.0 certification
```

The textual tree omits some cross-edges. The registry is authoritative.

## 5. WS00 — Governance, registries, and claim discipline

### Objective

Create the machine-checkable architecture governance substrate before implementation claims proliferate.

### Deliverables

- compatibility, evidence, claim, workstream, dependency, operator, domain, algorithm, verifier, and schema registries;
- registry canonicalization and digest rules;
- claims linter resolving README/docs statements to gates;
- discrepancy ledger schema;
- architecture decision record template;
- source-pin update protocol;
- dependency/layering constitution.

### Acceptance

- every registry parses and canonicalizes deterministically;
- unknown required fields fail closed;
- workstream graph is acyclic;
- every public present-tense capability claim resolves to evidence or is rejected;
- planning/target language is allowed but cannot render a shipped badge;
- a deliberate false claim fixture fails CI.

### Forbidden shortcuts

- prose-only claim status;
- wildcard evidence links;
- mutable registry semantics without version change;
- “implemented” inferred from a file or API name.

## 6. WS01 — Conformance laboratory foundation

### Dependencies

WS00.

### Objective

Build isolated upstream-oracle and FrankenSymPy runners before broad compatibility implementation.

### Deliverables

- immutable SymPy 1.14.0 environment manifest;
- separate-process fixture protocol;
- source/reflection inventory generator;
- observation envelope and comparator registry;
- mismatch minimizer and discrepancy writer;
- moving-head drift lane;
- exact environment/build capture.

### Acceptance

- oracle and candidate cannot import/share one another;
- fixtures reproduce identically across fresh processes;
- deliberate type, class, warning, printer, pickle, and held-form mismatches are detected;
- comparator mutation fixtures fail;
- moving-head results cannot certify the immutable profile.

### Forbidden shortcuts

- same-process object comparison as the only oracle;
- golden regeneration without profile review;
- API reachability counted as parity.

## 7. WS02 — Foundation types, schemas, budgets, and `Cx`

### Dependencies

WS00.

### Objective

Implement the typed execution and identity substrate shared by all native components.

### Deliverables

- distinct ID newtypes and canonical text/binary forms;
- `MathOutcome`/`ExecutionOutcome`;
- multidimensional hierarchical budgets and reservations;
- FrankenSymPy `Cx` wrapper over asupersync;
- deterministic random stream derivation;
- bounded schema envelopes and capability tokens;
- trace/receipt primitives.

### Acceptance

- type system rejects ID-category substitution in compile-fail tests;
- canonical IDs survive cross-process/cross-architecture fixtures;
- budget charges never reset on fallback;
- verifier reservation cannot be consumed by generators;
- deterministic lab runs reproduce traces;
- cancellation region leak detector remains zero.

### Forbidden shortcuts

- generic string IDs internally;
- wall-clock timeout as the only budget;
- direct detached spawning;
- timestamps or memory addresses in stable IDs.

## 8. WS03 — Exact arithmetic substrate

### Dependencies

WS02.

### Objective

Provide pure-Rust exact integer, rational, modular, and canonical arithmetic required by terms and verifiers.

### Deliverables

- canonical big integers and rationals;
- exact division/gcd/extended gcd;
- modular arithmetic, CRT, rational reconstruction;
- deterministic prime streams and primality primitives;
- multiple multiplication strategies with scalar reference;
- cancellation and limb/height accounting;
- bounded canonical serialization.

### Acceptance

- property/differential tests over broad sizes;
- algorithm-threshold boundary corpus;
- scalar and optimized decisions/values agree;
- malformed/oversized decoders fail before allocation;
- cancellation at every recursive/batch safe point;
- no C/C++ arithmetic FFI.

### Forbidden shortcuts

- machine-float exact intermediates;
- unchecked external arithmetic identity;
- optimized path without reference differential lane.

## 9. WS04 — Terms, domains, assumptions, and bindings

### Dependencies

WS02, WS03.

### Objective

Implement the Semantic Term DAG and immutable mathematical universe.

### Deliverables

- typed operator/payload term representation;
- deterministic interning and generation-checked handles;
- canonical `TermId` encoding;
- domain constructors and coercion graph;
- immutable assumptions contexts with four-way truth;
- capture-avoiding binders and alpha-equivalence;
- surface-protocol descriptors and lowering receipts;
- mutable snapshot descriptors.

### Acceptance

- cross-process stable term IDs;
- concurrent interning schedule exploration;
- commutative/noncommutative/domain invariants;
- `Unknown` never becomes false;
- contradiction policies pass generated context corpus;
- binder capture mutants are killed;
- term kernel builds without Python/persistence/network features.

### Forbidden shortcuts

- strings as IR;
- Python class or database row as semantic identity;
- global mutable assumptions registry;
- domain choice by incidental registration/hash order.

## 10. WS05 — Python compatibility shell vertical slice

### Dependencies

WS01, WS02, WS04.

### Objective

Prove that a real Python shell can preserve SymPy's object model while accelerating eligible regions natively.

### Initial surface

- `Basic`, numeric atoms, `Symbol`, `Dummy`;
- `Add`, `Mul`, `Pow`;
- `FunctionClass`, `Function`, undefined/applied functions;
- evaluated and held construction;
- equality, hashing, sorting, `args`, `func`;
- substitution/traversal/reconstruction;
- core assumptions;
- string/pretty/LaTeX printers;
- copy/pickle;
- one mutable matrix slice.

### Deliverables

- pure Python class shell and profile maps;
- contained safe CPython bridge;
- exact-class native fast-path checks;
- supervised callback lane;
- opaque user-node descriptors;
- lowering/lifting receipts;
- coexistable and drop-in package assembly skeletons.

### Acceptance

- full initial inventory and upstream differential suite;
- custom subclass/metaclass/hook corpus;
- held/evaluated corpus;
- hash-seed and pickle cross-process matrix;
- opaque nodes survive mixed native operations;
- no upstream SymPy import/fallback in product paths;
- shell-only fallback means FrankenSymPy's own implementation, never upstream execution.

### Forbidden shortcuts

- one opaque Rust extension class advertised as drop-in;
- canonicalizing held forms before shell observation;
- replacing arbitrary subclasses with built-ins;
- invoking Python callbacks from unsupervised native workers.

## 11. WS06 — Proof kernel and evidence system

### Dependencies

WS02, WS03, WS04.

### Objective

Create the small trusted kernel and non-inflatable result evidence contract.

### Deliverables

- typed claim schemas;
- proof-term checker;
- equality/congruence/substitution/assumption rules;
- evidence envelopes and verification receipts;
- separate candidate/verified namespaces;
- reference certificate-dispatch framework;
- mutation harness.

### Acceptance

- every proof constructor has positive/negative/adversarial fixtures;
- unknown claim/evidence/verifier schemas fail closed;
- stored verification flags are rejected;
- registered weakening mutants are killed;
- evidence non-conversion registry enforced;
- verifier dependencies exclude generator crates.

### Forbidden shortcuts

- search trace called proof;
- oracle parity called mathematical proof;
- user hook result silently upgraded to theorem;
- verifier importing generator implementation.

## 12. WS07 — Verified rewriting and simplification

### Dependencies

WS04, WS06.

### Objective

Build deterministic local rewriting and bounded goal-directed equality search with proof extraction.

### Deliverables

- versioned typed rewrite registry;
- side-condition obligations;
- AC/binder-aware matching where declared;
- deterministic canonical pipeline;
- bounded local e-graph;
- multi-objective extraction;
- proof path reconstruction;
- compatibility-specific surface transformation hooks.

### Acceptance

- every accepted rewrite has a verified derivation;
- unresolved side conditions remain guarded;
- e-graph cancellation/growth budgets pass adversarial corpus;
- rule-order and hash-order determinism tests;
- branch-cut/commutativity mutants killed;
- held surface forms preserved by profile policy.

### Forbidden shortcuts

- unconditional conditional rewrites;
- immortal global e-graph;
- “simplest” defined by one undocumented scalar score;
- e-class union without justification.

## 13. WS08 — Polynomial representations and arithmetic

### Dependencies

WS03, WS04, WS06.

### Objective

Provide adaptive exact polynomial rings and proof-checkable arithmetic.

### Deliverables

- dense, sparse, recursive, modular, and evaluation representations;
- canonical ring/monomial-order identities;
- addition/multiplication/division/pseudo-division;
- content, primitive, square-free, subresultant primitives;
- conversion receipts and invariant checks;
- polynomial identity verifier.

### Acceptance

- representation conversions round-trip exactly;
- dense/sparse/modular portfolios agree;
- polynomial identity certificates verify independently;
- degree/coefficient/term growth budget tests;
- unlucky-prime and zero-divisor adversaries;
- profile-correct `Poly` lowering/lifting fixtures when WS05 integrates.

### Forbidden shortcuts

- generic expression expansion as the polynomial engine;
- silent monomial-order/domain changes;
- sampled evaluations as exact identity without reconstruction/check.

## 14. WS09 — GCD, factorization, and certificates

### Dependencies

WS06, WS08, WS13 for final portfolio closure.

### Objective

Deliver the first crown-jewel proof-carrying algebra portfolio.

### Deliverables

- subresultant and modular GCD strategies;
- finite-field factorization;
- Hensel lifting and deterministic recombination;
- factorization/GCD certificate schemas;
- independent reference verifiers;
- checkpointable modular state;
- planner diagnostics and safe baseline.

### Acceptance

- exact product, normalization, multiplicity, and requested irreducibility obligations verified;
- deliberately omitted factor/incorrect irreducibility mutants rejected;
- multiple strategies race without completion-order publication;
- cancellation/checkpoint/resume across prime/recombination batches;
- parity-gated benchmark against upstream and scalar native lane;
- invalid remote candidate cannot pollute verified cache.

### Forbidden shortcuts

- product check presented as irreducibility proof;
- generator self-verification;
- lucky benchmark-only factor families.

## 15. WS10 — Exact linear algebra

### Dependencies

WS03, WS04, WS06, WS08, WS13 for portfolio closure.

### Objective

Implement dense/sparse exact matrix kernels and certificates.

### Deliverables

- fraction-free elimination;
- modular/p-adic solve, determinant, rank, nullspace;
- sparse/structured planning;
- exact decomposition and uniqueness/completeness certificates;
- symbolic parameter interpolation path;
- profile matrix shell integration.

### Acceptance

- `A*X=B`, decomposition, rank, determinant, nullspace claims independently checked;
- singular/rectangular/zero-dimensional edge corpus;
- modular reconstruction bound mutants rejected;
- memory/fill budgets and cancellation;
- live upstream/reference parity-gated benchmarks.

### Forbidden shortcuts

- floating rank used as exact rank;
- one solution called unique without rank obligation;
- dense materialization of every sparse input.

## 16. WS11 — Certified numerics and algebraic numbers

### Dependencies

WS03, WS04, WS06.

### Objective

Provide directed-rounding balls, exact algebraic numbers, and certified root/isolation claims.

### Deliverables

- real/complex ball primitives;
- adaptive precision engine;
- algebraic number fields and embeddings;
- exact real root isolation/refinement;
- enclosure and root-completeness certificates;
- branch/singularity-aware evaluation;
- exact recognition checker.

### Acceptance

- every certified enclosure contains independent high-precision/reference values;
- directed-rounding mutants killed;
- algebraic equality/order/minimal-polynomial corpus;
- all-root completeness and multiplicity adversaries;
- recognition candidates require exact verification;
- NaN/infinity/branch-cut profile tests.

### Forbidden shortcuts

- ordinary float plus guessed epsilon called certified;
- approximate value as algebraic identity;
- reported roots checked without completeness.

## 17. WS12 — Differentiation and symbolic compilation

### Dependencies

WS04, WS05, WS06, WS07, WS08, WS10, WS11, WS13.

### Objective

Build proof-producing differentiation and verified residual/Jacobian/Hessian compilation into FrankenNumPy/FrankenSciPy targets.

### Deliverables

- structural derivative proof rules;
- high-order/multivariate DAG dynamic programming;
- sparse Jacobian/Hessian analysis and coloring;
- target-neutral numeric IR;
- CSE and target-aware algebraic transforms with receipts;
- generated evaluator plus scalar/exact/certified reference lane;
- custom-function opaque callback boundary.

### Acceptance

- derivative proof replay;
- custom `_eval_derivative` provenance preserved;
- sparse structure agrees with exact symbolic dependencies;
- generated residual/Jacobian values match exact/certified lane;
- domain/branch guards generated and tested;
- cross-target deterministic source/object digest where promised;
- end-to-end hero workload gates.

### Forbidden shortcuts

- finite differences as exact derivatives;
- CSE/rewrites without semantic proof;
- compiling opaque Python behavior as if pure/native.

## 18. WS13 — Structured portfolios, cancellation, and replay

### Dependencies

WS02, WS04, WS06; integrates WS07–WS12 incrementally.

### Objective

Implement the common proof-carrying speculative execution fabric.

### Deliverables

- diagnostics/evidence-vector framework;
- versioned loss policies and decision cards;
- owned strategy/verifier region topology;
- protected verifier budgets;
- two-phase candidate publication;
- deterministic and replay modes;
- typed continuations;
- e-process/conformal operational monitors.

### Acceptance

- injected cancellation at every publication boundary;
- zero controlled orphan tasks;
- candidate rejection triggers fallback correctly;
- completion/arrival order cannot change strict result;
- replay bundle reproduces terminal digest;
- monitor alarms change rollout policy only, never evidence class;
- selector cannot learn from unverified success.

### Forbidden shortcuts

- first completed candidate wins;
- detached speculative work;
- posterior/e-process promoted to proof;
- budget reset on fallback.

## 19. WS14 — Agent protocol and semantic workspaces

### Dependencies

WS00, WS02, WS04, WS06, WS13.

### Objective

Expose structured symbolic state for agents and collaborative derivation branches.

### Deliverables

- NDJSON/RPC schemas and streaming terminal semantics;
- term/claim/evidence/receipt introspection;
- semantic patches;
- branch fork/review/merge/rebase;
- counterexample bundles;
- bounded work packets and Beads conversion tooling;
- deterministic agent-session replay.

### Acceptance

- unknown schema/evidence fields fail closed;
- semantic patch idempotence/conflict corpus;
- same-print/different-domain merge rejection;
- candidate edges never become verified through merge;
- proof pagination stable across rebuild;
- transcript-free replay reconstructs mathematical state;
- malformed/oversized streams rejected before publication.

### Forbidden shortcuts

- strings or transcript as authoritative state;
- natural-language claim of completion as a gate;
- remote client writing verified edges directly.

## 20. WS15 — Persistence, checkpoints, and RaptorQ repair

### Dependencies

WS02, WS04, WS06, WS13.

### Objective

Provide optional durable workspaces and recoverable expensive computation without making storage authoritative for truth.

### Deliverables

- storage-neutral append-only ledger traits;
- FrankenSQLite adapter;
- verified/candidate cache separation;
- typed checkpoints/continuations;
- two-phase artifact publication;
- RaptorQ repair envelopes and scrubber;
- GC/retention and schema migration.

### Acceptance

- crash injection at every publication boundary;
- repaired bytes pass digest/schema before resume;
- proof verification remains separate and can still fail after byte repair;
- stale universe/cache entry rejected;
- candidate `verified=true` poisoning rejected;
- fresh-process deterministic resume;
- persistence-disabled run has identical semantic result.

### Forbidden shortcuts

- database row identity for terms/proofs;
- process memory dump as checkpoint;
- RaptorQ decode called integrity or truth;
- database transaction in every algebraic hot path.

## 21. WS16 — Remote workers and graph indexing

### Dependencies

WS14, WS15, WS06, WS13.

### Objective

Scale search and collaborative discovery while retaining local verification and authoritative ledger semantics.

### Deliverables

- content-addressed work packets and leases;
- duplicate/late/byzantine response handling;
- local verification coordinator;
- FrankenGraphDB derivation/dependency projection;
- index rebuild and projection-version protocol;
- capability/privacy controls.

### Acceptance

- wrong claim/context/domain response rejected;
- duplicate/late responses cannot double-publish;
- worker cannot write branch head or verified cache;
- graph index deletion/rebuild preserves authoritative answers;
- graph reachability never satisfies proof APIs;
- local-only sensitive objects never leave coordinator;
- remote cancellation revokes publication rights and drains controlled tasks.

### Forbidden shortcuts

- worker majority/reputation as proof;
- graph database as term/proof source of truth;
- content ID used as authorization token.

## 22. WS17 — Gröbner bases and ideal algebra

### Dependencies

WS06, WS08, WS10, WS13.

### Objective

Deliver verifier-backed Gröbner/ideal computation across deterministic, sparse, and modular strategies.

### Deliverables

- Buchberger reference;
- F4-style sparse linear algebra;
- signature/F5 and modular research lanes;
- FGLM/order conversion;
- ideal membership and S-pair certificates;
- incremental workspace updates;
- checkpointable pair/matrix state.

### Acceptance

- basis and ideal-membership certificates independently verified;
- reduced/minimal claims checked separately;
- omitted S-pair and wrong monomial-order mutants killed;
- deterministic pair/tie order;
- cancellation/growth/adversarial benchmarks;
- parity-gated corpus over multiple domains/orders.

## 23. WS18 — Integration, limits, series, and transforms

### Dependencies

WS04, WS06, WS07, WS08, WS11, WS13, WS17 where needed.

### Objective

Implement branch/domain/convergence-aware calculus portfolios with honest evidence classes.

### Deliverables

- rational/algebraic/rule/heuristic integration strategies;
- exact antiderivative verification and definite-integral obligations;
- formal and analytic series separation;
- limits/asymptotic engine;
- sums/products/creative telescoping;
- integral transforms and convergence-region metadata;
- certified numeric fallback.

### Acceptance

- derivative-only verifier cannot certify unsupported definite/completeness claims;
- branch/singularity/convergence adversarial corpus;
- formal-series versus analytic-convergence distinction tested;
- conditional results preserve obligations;
- “not found” cannot become “nonexistent”;
- compatibility forms/types/printers pass profile fixtures.

## 24. WS19 — Solvers, sets, logic, ODE, and PDE

### Dependencies

WS04, WS06, WS08, WS10, WS11, WS13, WS17, WS18 selectively.

### Objective

Provide typed solution sets and completeness-aware logical/mathematical solvers.

### Deliverables

- polynomial/transcendental equation portfolios;
- inequalities/CAD/real sign methods;
- SAT/logic with checkable UNSAT evidence;
- exact/lazy set operations;
- Diophantine and number-theory solvers;
- ODE classifications/solution verification;
- selected PDE solution families and residual checks.

### Acceptance

- every solution satisfies original constraints after denominator/branch filtering;
- completeness status explicit and verified when claimed;
- SAT model/UNSAT proof checks;
- excluded/extraneous roots corpus;
- singular ODE solution and boundary-condition tests;
- one PDE solution never implied generality.

## 25. WS20 — Structured mathematics domains

### Dependencies

WS04, WS06, WS07, WS10, WS11, WS13, WS19 selectively.

### Objective

Build geometry, tensors, units, statistics, and physics as compositions over the common semantic/evidence substrate.

### Deliverables

- permutation-group tensor canonicalization;
- exact/certified geometry predicates;
- dimensions/units/conversion registries;
- random variables/distributions/expectation objects;
- mechanics/quantum/vector/control modules;
- convention/source provenance.

### Acceptance

- no factorial default tensor canonicalization;
- degeneracy and dimensional inconsistency corpus;
- distributional/formal/numeric result classes separated;
- physical convention/constant edition included in context;
- profile classes/printers/serialization conform.

## 26. WS21 — Compatibility and ecosystem closure

### Dependencies

WS01, WS05 and every surface-bearing workstream included in the target profile.

### Objective

Close the complete SymPy 1.14.0 profile rather than shipping an ambiguous “mostly compatible” badge.

### Deliverables

- complete public/semi-private inventory;
- applicable upstream suite;
- generated differential and metamorphic corpus;
- custom subclass/held/mutable/printer/pickle suites;
- ecosystem packages/notebooks;
- discrepancy ledger closure;
- no-oracle-runtime inspection.

### Acceptance

All gates in `COMPATIBILITY_CONTRACT.md` pass on one commit/build matrix. Every exclusion is explicit and nonblocking under the immutable profile policy. No hidden upstream runtime dependency exists.

### Forbidden shortcuts

- blended behavior matching no release;
- deleting difficult tests;
- mathematical equivalence excusing object-model drift;
- preview called certified.

## 27. WS22 — Performance and architecture optimization

### Dependencies

Begins after WS03/WS04 reference lanes; closes only after relevant semantic gates.

### Objective

Create radical performance gains without weakening compatibility, evidence, cancellation, or durability.

### Deliverables

- profile-guided representation/algorithm thresholds;
- SIMD/parallel kernels behind scalar reference lanes;
- memory locality, arena, cache, and compile-time improvements;
- batch/agent throughput optimization;
- architecture dispatch for Apple Silicon and high-core x86;
- parity-gated benchmark evidence.

### Acceptance

- live incumbent measured in same invocation;
- semantic parity before sample admission;
- protected tail/memory/proof/cancellation metrics;
- hidden/generated holdouts defeat benchmark hard-coding;
- stable IDs/evidence identical across optimized/reference lanes;
- optimization unsafe islands, if any, separately audited.

## 28. WS23 — Packaging, release, and 1.0 certification

### Dependencies

WS00, WS21, WS22 and all features claimed by the release.

### Objective

Publish coexistable and certified drop-in artifacts with one reproducible evidence bundle.

### Deliverables

- Rust crates and CLI;
- `frankensympy` Python distribution;
- `frankensympy-dropin` for certified profiles;
- platform wheels and Wasm subset;
- SBOM/build provenance;
- compatibility, proof, runtime, security, repair, ecosystem, and benchmark reports;
- signed/content-addressed release manifests and optional repair sidecars.

### Acceptance

- full release gate matrix on the exact release commit;
- all public claims resolve to same-commit artifacts;
- installation conflict/coexistence behavior tested;
- no upstream SymPy fallback/dependency;
- fresh-process proof/replay bundles verify;
- rollback/advisory/invalidation procedures tested.

## 29. Critical path to the first hero workload

The shortest architecture-proving path is:

```text
WS00 → WS02 → WS03 → WS04 → WS06 → WS08
             ↘ WS01 → WS05 ↗       ↘ WS09
WS02/04/06 → WS13
WS04/05/06/07/08/10/11/13 → WS12
WS13 → WS15
WS00/02/04/06/13 → WS14
```

The first hero workload closes only when WS05, WS06, WS08, WS09, WS12, WS13, WS14, and the required WS15 slice pass together. It is defined in `FIRST_IMPLEMENTATION_CAMPAIGN.md`.

## 30. Parallelism plan

After WS00/WS02 interfaces freeze sufficiently:

- WS01 proceeds independently on oracle tooling;
- WS03 arithmetic and WS05 shell prototyping can proceed in parallel, though WS05 native closure waits on WS04;
- WS06 proof schemas can proceed beside WS04 term/domain implementation;
- polynomial, certified numeric, and protocol scaffolding can proceed in parallel after semantic contracts;
- verifier teams remain separate from optimizing generator teams;
- persistence adapters can be built against traits before algorithm-specific checkpoints;
- breadth workstreams begin only after the first vertical slice validates the shell/kernel/proof/runtime composition.

Parallelism is used to shorten the critical path, not to create unchecked API surface.

## 31. Milestone exit gates

### M0 — Planning substrate

WS00 closes; architecture docs/registries agree; every capability is still explicitly planned.

### M1 — Native semantic nucleus

WS02–WS04 and the initial WS06 kernel close; stable terms/domains/contexts/proofs work without Python.

### M2 — Python object-model slice

WS01 and initial WS05 close for the first profile surface, including arbitrary subclasses and held forms.

### M3 — Proof-carrying algebra portfolio

WS08, WS09, and the relevant WS13 slice close with exact certificates, cancellation, and parity-gated benchmarks.

### M4 — Certified Jacobian hero pipeline

Initial WS07, WS10, WS11, and WS12 compose with M2/M3 and generate verified numeric programs.

### M5 — Durable/agent-native fabric

Initial WS14–WS16 slices demonstrate semantic branching, checkpoint repair/resume, untrusted workers, and rebuildable graph indexing.

### M6 — Compatibility expansion

Large portions of algebra/calculus/solver/structured surfaces are profile-conformant with a shrinking public discrepancy ledger.

### M7 — Breadth and optimization closure

WS17–WS22 target scope closes, ecosystem corpus is green, and leapfrog performance claims have parity-gated evidence.

### M8 — Certified drop-in 1.0

WS23 closes for at least one immutable SymPy profile and supported platform matrix.

## 32. Beads conversion template

Each workstream is decomposed into Beads only after every task record includes:

```text
Task ID and workstream
Objective / non-goals
Exact dependency IDs
Owned files/crates/registries
Inputs and immutable universe
Implementation deliverable
Independent verifier/gate owner
Acceptance commands
Required unit/property/differential/metamorphic/adversarial tests
Benchmark obligations and live incumbent
Discrepancy/claim-registry effects
Cancellation/resource/failure behavior
Forbidden shortcuts
Artifacts proving closure
```

A broad task such as “implement integration” is invalid. It must be split by bounded algorithm/certificate/profile surface with objective closure.

## 33. Graph-change protocol

To add or alter a workstream:

1. update the machine registry;
2. prove acyclicity;
3. explain changed critical path and milestone effects;
4. update affected claims/risks;
5. identify migration for already-issued work packets;
6. preserve retired IDs as tombstones;
7. commit graph and prose changes together or in immediately adjacent verified commits.

## 34. Program-level forbidden shortcuts

- parallelizing thousands of API stubs before the vertical architecture works;
- closing a generator without its independent verifier/gates;
- moving a hard requirement to “later” without graph/claim/risk updates;
- declaring a milestone by percentage complete;
- allowing cyclic “temporary” dependencies;
- measuring performance before semantic admission;
- letting persistence or remote work become mandatory for local correctness;
- treating a planning document as implementation evidence;
- assigning one agent both the feature and sole authority to weaken its acceptance gate;
- merging work by prose confidence rather than artifacts.
