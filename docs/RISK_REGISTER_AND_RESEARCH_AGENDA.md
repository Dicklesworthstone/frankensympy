# FrankenSymPy risk register and research agenda

**Status:** normative planning input  
**Scope:** existential risks, technical risks, trigger metrics, mitigations, residual uncertainty, open architectural decisions, and research program

## 1. Purpose

A radical symbolic system can fail by being unsound, incompatible, slow, unmaintainable, or merely impressive in demos. This register treats those as different failure modes with different evidence and response plans.

Risk status is updated through commits and gate artifacts. Optimistic prose does not lower risk.

## 2. Severity and likelihood

### Severity

- **S5 — Existential:** invalidates the core architecture, mathematical trust, or drop-in objective.
- **S4 — Critical:** blocks a major milestone or public claim.
- **S3 — High:** materially harms breadth, performance, safety, or maintainability.
- **S2 — Moderate:** bounded subsystem or schedule impact.
- **S1 — Low:** localized inconvenience or polish debt.

### Likelihood

- **L5 — Expected:** likely without active mitigation.
- **L4 — Probable:** substantial chance.
- **L3 — Plausible:** meaningful but uncertain chance.
- **L2 — Unlikely:** possible, with known controls.
- **L1 — Remote:** requires unusual conditions.

No numeric product is used as a substitute for judgment. S5 risks receive architecture-level attention regardless of estimated likelihood.

## 3. R-001 — Python object-model boundary is incomplete

**Severity:** S5  
**Likelihood:** L4 before the first vertical slice

### Failure mode

A real SymPy behavior involving metaclasses, dynamic classes, `__new__`, `_eval_*`, constructor postprocessors, mutable objects, held forms, pickles, or ecosystem introspection cannot be preserved by the proposed shell/native boundary without either disabling native acceleration broadly or changing public behavior.

### Triggers

- minimized differential fixture has no representable surface descriptor;
- lower/lift requires executing unknown hooks in an unsupervised native path;
- `func(*args)` or pickle reconstruction cannot preserve class identity;
- native handle caching becomes invalid under legal subclass mutation/override behavior;
- coexistable and drop-in package builds require incompatible class implementations.

### Mitigations

- shell remains authoritative for Python identity;
- exact-class fast-path checks;
- conservative opaque nodes and shell-only paths;
- surface/semantic/provenance separation;
- custom-subclass corpus before broad API work;
- architecture review on every unrepresentable fixture;
- no requirement that every operation lower natively.

### Closure evidence

WS05 object-model gate and ecosystem subclass corpus. One passing demo is insufficient.

### Residual risk

The architecture may remain correct but native acceleration coverage may be lower than hoped for highly dynamic workloads. That is preferable to false drop-in claims.

## 4. R-002 — Compatibility scope is larger than implementation capacity

**Severity:** S5  
**Likelihood:** L4

### Failure mode

SymPy's public and widely used semi-private surface, historical quirks, optional dependencies, printers, assumptions, mutable classes, and ecosystem use make full profile closure prohibitively large.

### Triggers

- discrepancy arrival rate does not decline with surface expansion;
- ecosystem corpus exposes pervasive undocumented coupling;
- profile maintenance consumes more effort than native capability development;
- release exclusions become broad enough to undermine “drop-in.”

### Mitigations

- immutable profiles rather than a moving target;
- generated inventory and differential fixture machinery;
- repetitive shell declarations generated from reviewed manifests;
- prioritize central object-model contracts early;
- profile behavior tables where safe;
- transparent preview package before drop-in certification;
- multiple contributors/agents working from bounded work packets.

### Closure evidence

WS21 complete inventory, upstream suite, generated differential, ecosystem corpus, and closed blocking discrepancy ledger.

### Residual risk

Supporting additional SymPy releases remains ongoing work. Certification applies only to named profiles.

## 5. R-003 — Pure-Rust arbitrary-precision substrate underperforms or is incorrect

**Severity:** S5  
**Likelihood:** L3

### Failure mode

Avoiding GMP/FLINT/Arb FFI creates a large performance and correctness burden in big integers, modular arithmetic, polynomial kernels, algebraic numbers, and ball arithmetic.

### Triggers

- reference arithmetic fails adversarial/property corpora;
- performance remains structurally noncompetitive at relevant sizes;
- optimized multiplication/rounding introduces architecture-dependent results;
- foundational external crate cannot meet determinism, cancellation, or serialization needs;
- safe Rust implementation requires unacceptable memory overhead.

### Mitigations

- scalar reference first;
- strategy portfolio by size/architecture;
- narrow replaceable arithmetic façade;
- broad independent test vectors and mutation testing;
- architecture-specific optimization only after exact parity;
- collaboration with other Franken numeric projects;
- explicit provisional foundational dependency decision if necessary.

### Closure evidence

WS03/WS11 gates and paired benchmarks across Apple Silicon/high-core x86.

### Residual risk

The system may initially win at high-level DAG/batch/parallel workloads before matching specialized low-level libraries on every arithmetic kernel.

## 6. R-004 — Proof verifier is not meaningfully independent

**Severity:** S5  
**Likelihood:** L3

### Failure mode

Generator and verifier share enough code or assumptions that one defect causes both to accept a false result, undermining the proof-carrying architecture.

### Triggers

- verifier depends on generator crate;
- certificate checking repeats the generator's entire algorithm;
- shared optimized kernel dominates both paths without a reference lane;
- mutation survives because both sides share the mutated helper;
- “cross-check” strategies share the same faulty normalization.

### Mitigations

- dependency-level generator/verifier separation;
- simple reference verifiers;
- claim-specific certificates;
- negative/adversarial/mutation corpora;
- independent mathematical or brute-force oracles for small cases;
- trust manifest naming shared components;
- high-value verifier review ownership separate from generator ownership.

### Closure evidence

WS06 and family-specific mutation gates.

### Residual risk

Some arithmetic primitives remain shared trusted dependencies. Their risk is addressed by reference implementations and independent test vectors, not denied.

## 7. R-005 — Evidence classes are inflated by product pressure

**Severity:** S5  
**Likelihood:** L3

### Failure mode

Heuristic candidates, oracle parity, numeric agreement, e-process confidence, or repaired artifacts are marketed as proofs or exact results.

### Triggers

- native API returns one result variant for candidate and verified values;
- README language exceeds claims registry evidence;
- selector posterior appears in acceptance code;
- compatibility match is cited as mathematical correctness;
- RaptorQ decode record is treated as object validity;
- “proof-producing” claim has no live verifier/mutation gate.

### Mitigations

- non-inflatable evidence registry;
- claims linter and same-commit artifact resolution;
- separate candidate/verified caches and terminal variants;
- prohibited-promotion tests;
- public discrepancy/claim state;
- incident severity S5 for unsound acceptance.

### Closure evidence

WS00/WS06/WS23 claim closure.

## 8. R-006 — Assumptions or branch logic makes transformations unsound

**Severity:** S5  
**Likelihood:** L4

### Failure mode

Rules valid only for positive reals, nonzero denominators, integer parameters, or specific analytic branches are applied unconditionally.

### Triggers

- `Unknown` interpreted as false/true;
- context omitted from cache key;
- branch policy omitted from claim/term transformation;
- contradictory assumptions explode into arbitrary conclusions;
- custom `_eval_is_*` hook silently receives theorem status;
- sampled complex values disagree with rewrite.

### Mitigations

- immutable context IDs and four-way internal truth;
- guarded rewrite relations;
- explicit branch policy;
- side-condition proof obligations;
- paraconsistent contradiction option;
- branch-cut/adversarial generated corpus;
- cache universe completeness.

### Closure evidence

WS04, WS07, WS11, WS18 mutation and differential gates.

## 9. R-007 — Canonicalization destroys surface fidelity

**Severity:** S5  
**Likelihood:** L4 without three-graph separation

### Failure mode

Native normalization erases held argument order, multiplicity, custom class identity, binder spelling, or printer/pickle behavior required by the compatibility profile.

### Triggers

- one graph used for both shell and semantic terms;
- `evaluate=False` round trip changes `args`;
- lowering modifies shell object/hash before the call contract permits it;
- alpha-equivalence substituted for Python equality;
- result lifting invokes constructors that reevaluate unexpectedly.

### Mitigations

- authoritative Surface Object Graph;
- explicit lowering/lifting receipts;
- profile-aware lift policies;
- separate stable IDs and Python hashes;
- held-form corpus;
- constructor side-effect observation.

### Closure evidence

WS05 object-model gate.

## 10. R-008 — Structured-concurrency guarantees fail at Python/foreign boundaries

**Severity:** S4  
**Likelihood:** L4

### Failure mode

A non-cooperative Python hook or third-party extension blocks cancellation, retains work, or mutates state after the native region has returned.

### Triggers

- callback executes on unsupervised native worker;
- return occurs while callback-owned activity continues;
- product claims universal cancellation latency;
- callback reentrancy bypasses nested budgets;
- forceful interruption corrupts interpreter state.

### Mitigations

- supervised callback lane;
- explicit non-cooperative boundary outcome;
- optional separate-process sandbox;
- pre/post callback cancellation checks;
- nested budgets and reentrancy tracking;
- narrow cancellation claims.

### Closure evidence

WS05/WS13/WS02 hook delay/reentrancy/cancellation corpus.

### Residual risk

Strict compatibility can expose upstream-style arbitrary Python execution. Hardened mode isolates or disables it; the native core cannot guarantee foreign code behavior.

## 11. R-009 — Portfolio overhead overwhelms useful computation

**Severity:** S4  
**Likelihood:** L3

### Failure mode

Diagnostics, speculative strategies, proof generation, and verification cost more than a simple deterministic algorithm, especially for small calls.

### Triggers

- strategy launch occurs below crossover size;
- proof/receipt size dominates result;
- Python boundary plus portfolio latency loses broadly on scalar calls;
- verifier reservation wastes large budgets after trivial candidates;
- adaptive policy oscillates.

### Mitigations

- cheap safe baseline and size-aware eligibility;
- single-strategy deterministic path for small instances;
- cached verified results and compact proof macros;
- decision-card cost includes verifier/launch overhead;
- paired crossover benchmarks;
- batch compilation/amortization focus.

### Closure evidence

WS13/WS22 workload-stratified benchmarks.

## 12. R-010 — Determinism conflicts with maximum parallel performance

**Severity:** S4  
**Likelihood:** L3

### Failure mode

Fixed reduction/tie/launch policies reduce throughput, or high-performance completion-order behavior changes output forms and replay.

### Triggers

- semantic winner depends on task arrival;
- deterministic data structures dominate runtime;
- architecture-specific paths produce different forms/certificates;
- replay cannot reproduce adaptive launch decisions;
- numeric reductions differ across core counts.

### Mitigations

- separate strict, replay, and latency-adaptive modes;
- verifier-governed acceptance plus deterministic extraction tie breaks;
- counter-based random streams;
- recorded adaptive decisions;
- semantic versus byte determinism explicitly scoped;
- scalar/reference parity.

### Closure evidence

WS13/WS22 deterministic and performance matrices.

## 13. R-011 — Persistence contaminates the algebraic hot path

**Severity:** S4  
**Likelihood:** L3

### Failure mode

Database transactions, row identities, or durable cache lookups become required for ordinary term construction and rewriting, creating latency, coupling, and correctness risk.

### Triggers

- `TermId` depends on row ID;
- local ephemeral mode cannot run the same computation;
- every rewrite issues storage I/O;
- persistence failure changes mathematical result rather than execution outcome;
- graph index becomes required for proof traversal.

### Mitigations

- storage-neutral optional traits at L5;
- in-memory authoritative semantic kernel;
- content IDs independent of storage;
- ephemeral/persistent equivalence tests;
- derived indexes rebuildable.

### Closure evidence

WS15/WS16 disable/rebuild/fault gates.

## 14. R-012 — RaptorQ adds complexity without earned value

**Severity:** S3  
**Likelihood:** L3

### Failure mode

Repair encoding/metadata/storage overhead exceeds recomputation value, or the feature expands the trusted base and creates false durability claims.

### Triggers

- repair used on tiny/disposable artifacts;
- no artifact-value policy;
- encode/decode overhead omitted from benchmarks;
- repaired bytes not digest-checked;
- durability badge exists without loss/crash matrix;
- source and repair symbols share one failure domain.

### Mitigations

- expected-loss/value policy;
- selective artifact classes;
- strict decode → digest → schema → verification sequence;
- measured recomputation alternative;
- failure-domain-aware placement;
- explicit `planned` status until end-to-end repair gates.

### Closure evidence

WS15 fault/benchmark artifacts.

## 15. R-013 — e-process/conformal monitoring is statistically misapplied

**Severity:** S4  
**Likelihood:** L3

### Failure mode

Monitoring assumptions are violated, repeated resets/subgroup selection inflate alarms, or monitor outputs are treated as mathematical correctness.

### Triggers

- no stated filtration/null/exchangeability assumptions;
- adaptive subgroup mining without correction;
- monitor reset after unfavorable evidence;
- alarm directly promotes/rejects a mathematical candidate;
- telemetry includes selectively reported outcomes.

### Mitigations

- monitor registry with assumptions/actions/reset policy;
- use only for operational streams and rollout control;
- all outcomes, including failures/refusals, logged under privacy policy;
- shadow baselines and subgroup governance;
- prohibited promotion enforcement.

### Closure evidence

WS13 monitor simulation and claims tests.

## 16. R-014 — Mutable compatibility objects violate native immutability assumptions

**Severity:** S4  
**Likelihood:** L4

### Failure mode

Matrix/container mutation invalidates native handles, cache entries, snapshots, or aliases silently.

### Triggers

- mutable object interned as immutable term;
- mutation generation omitted from snapshot;
- native result writes back unexpectedly;
- shallow/deep copy behavior diverges;
- concurrent mutation races with compilation.

### Mitigations

- explicit mutable cells and immutable snapshots;
- generation validation;
- profile-correct alias/copy semantics;
- no cross-boundary write-back without mutating API contract;
- mutable-object conformance corpus.

### Closure evidence

WS05/WS10/WS12 mutation and snapshot gates.

## 17. R-015 — Serialization and pickle compatibility becomes a security or fidelity trap

**Severity:** S4  
**Likelihood:** L4

### Failure mode

Native canonical formats miss required profile behavior, or Python pickle support introduces hidden code execution and module/class reconstruction drift.

### Triggers

- network protocol accepts pickle implicitly;
- dynamic/local classes reconstructed incorrectly;
- module paths differ between coexistable/drop-in builds;
- unknown native schema fields silently ignored at proof boundaries;
- canonical decoder allocates from untrusted lengths.

### Mitigations

- separate safe native formats from explicit unsafe pickle capability;
- profile-specific module assembly;
- cross-process/protocol pickle matrix;
- bounded canonical decoders;
- hardened mode disables pickle.

### Closure evidence

WS05/WS14/WS21 security and compatibility gates.

## 18. R-016 — Generated symbolic-to-numeric code is subtly wrong

**Severity:** S5  
**Likelihood:** L3

### Failure mode

CSE, branch lowering, sparse layout, target function mapping, or floating transformations change semantics or derivatives.

### Triggers

- generated evaluator disagrees with exact/certified reference;
- domain guard omitted;
- target lacks a function/rounding convention and silently substitutes;
- sparse index mapping changes variable order;
- optimizer applies unsafe algebraic identity.

### Mitigations

- proof-producing transformation pipeline;
- source `TermId`/context/target ABI receipts;
- exact/certified reference evaluator and generated test vectors;
- explicit branch/domain guards;
- target-specific conformance profiles;
- fallback to less optimized proven IR.

### Closure evidence

WS12 hero and target corpus.

## 19. R-017 — Algorithm selector reward hacking or benchmark overfitting

**Severity:** S4  
**Likelihood:** L3

### Failure mode

The planner improves published metrics by learning benchmark identities, excluding failures, weakening evidence, or exploiting cache/mode asymmetries.

### Triggers

- benchmark IDs/features accessible to selector;
- training/evaluation corpus overlap;
- only successful outcomes reported;
- warm candidate versus cold incumbent;
- selector can alter comparator/evidence/durability;
- impressive aggregate hides subgroup regressions.

### Mitigations

- hidden/generated holdouts;
- parity gate outside selector control;
- immutable release loss/evidence policy;
- live paired incumbent;
- outcome-mix and subgroup reporting;
- e-process drift/regression monitoring;
- decision-card audits.

### Closure evidence

WS22 anti-reward-hacking suite.

## 20. R-018 — Expression and proof denial of service

**Severity:** S5 for services, S4 for libraries  
**Likelihood:** L5 without controls

### Failure mode

Small hostile inputs cause unbounded expansion, matching, solver enumeration, proof size, output printing, or memory use.

### Triggers

- no preflight/reservation before expansion;
- output/printer budgets absent;
- e-graph/global caches grow indefinitely;
- proof verifier trusts declared sizes;
- repeated fallback resets budget;
- one tenant starves verifier capacity.

### Mitigations

- multidimensional hierarchical budgets;
- growth estimates and reservations;
- bounded local search;
- proof/output limits;
- admission/fairness policy;
- resumable/refused outcomes;
- adversarial workload suite.

### Closure evidence

WS02/WS07/WS13/security gates.

## 21. R-019 — Distributed execution leaks data or accepts malicious work

**Severity:** S5  
**Likelihood:** L3

### Failure mode

Sensitive formulas leave local policy boundaries, workers return malicious payloads, or content-addressed IDs leak cross-tenant equality.

### Triggers

- packet includes unnecessary objects;
- worker can browse object store;
- response causes code execution or large allocation;
- worker signature/majority substitutes for verification;
- global unkeyed dedup across tenants;
- late revoked worker publishes.

### Mitigations

- least-privilege packet capabilities;
- local-only default and privacy-scoped stores/IDs;
- bounded canonical response decoding;
- local mathematical verification;
- lease/revocation and two-phase publication;
- no raw formulas in telemetry.

### Closure evidence

WS16 security/adversarial corpus.

## 22. R-020 — Franken-suite integrations create version and layering coupling

**Severity:** S3  
**Likelihood:** L4

### Failure mode

Asupersync, FrankenSQLite, FrankenGraphDB, FrankenNumPy, or FrankenSciPy APIs evolve, forcing core semantic changes or making optional adapters mandatory.

### Triggers

- core crates import integration-specific types;
- profile IDs include adapter versions unnecessarily;
- adapter feature changes `TermId` or proof semantics;
- source project claim assumed without live-path integration evidence;
- synchronized release required for unrelated local computations.

### Mitigations

- narrow traits and adapters at L4/L5;
- pinned source revisions per release;
- adapter contract/fault tests;
- semantic identity independent of integration;
- optional feature disable/equivalence gates;
- source re-audit protocol.

### Closure evidence

Layering CI and WS12/WS15/WS16 integration tests.

## 23. R-021 — Dependency minimalism delays foundational work excessively

**Severity:** S3  
**Likelihood:** L3

### Failure mode

Reimplementing every infrastructural component consumes effort better spent on symbolic innovation, while provisional dependencies become taboo rather than evaluated risks.

### Triggers

- months spent duplicating safe mature scaffolding with no strategic gain;
- custom parser/serializer/CPython layer creates more unsafe risk;
- architecture stalls awaiting ideal bigint/ball implementation;
- dependency decisions lack explicit criteria.

### Mitigations

- foundational exception process;
- containment behind removable façades;
- exact feature/transitive/unsafe review;
- provisional status and exit gate;
- strategic ownership focused on arithmetic, semantics, proof, and execution where differentiation matters.

### Closure evidence

Dependency registry and architecture review, not ideology alone.

## 24. R-022 — Build time and crate fragmentation impede development

**Severity:** S3  
**Likelihood:** L3

### Failure mode

A vast crate graph and generated profile surface create slow builds, complex features, and difficult agent navigation.

### Triggers

- crate-per-file proliferation;
- broad façade changes rebuild entire workspace;
- generated code dominates compiler memory;
- feature combinations explode;
- work packets routinely touch many layers.

### Mitigations

- split crates only for trust/dependency/ownership/compile boundaries;
- stable narrow interfaces;
- codegen tables/data over giant monomorphized code where appropriate;
- change-aware CI plus full release matrix;
- compile-time benchmarks;
- architecture lint for unnecessary edges.

### Closure evidence

WS22 compile-time and dependency metrics.

## 25. R-023 — Documentation outruns implementation reality

**Severity:** S4  
**Likelihood:** L5 in an ambitious plan

### Failure mode

Future-state architecture is read or advertised as shipped capability.

### Triggers

- present-tense README statements without claim IDs;
- badges before gate artifacts;
- plan status omitted;
- old target documents remain prominent after design changes;
- release claims reference different commits.

### Mitigations

- explicit current-status block;
- claims registry/linter;
- planning versus runtime versus certification language;
- same-commit artifact resolution;
- source/docs re-audit on releases;
- public discrepancy and implementation status.

### Closure evidence

WS00/WS23 claims closure.

## 26. R-024 — Agent swarm creates internally inconsistent architecture

**Severity:** S4  
**Likelihood:** L4

### Failure mode

Parallel agents implement incompatible IDs, contexts, evidence classes, schemas, or duplicate registries; merge by textual convenience hides semantic conflicts.

### Triggers

- tasks lack frozen universe/dependencies;
- multiple writers edit structural graph/registry concurrently;
- one agent implements feature and weakens gate;
- same term/claim schemas fork;
- completion asserted without acceptance artifacts.

### Mitigations

- workstream DAG and Beads conversion gate;
- single-writer structural registry changes;
- semantic workspaces/patches;
- independent gate owner;
- immutable schema/version IDs;
- machine acceptance commands and artifacts.

### Closure evidence

WS14 agent workflow and actual campaign execution history.

## 27. Open decision D-001 — First supported CPython/platform matrix

### Question

Which exact CPython versions and platform tags define the first immutable profile build matrix?

### Default until decided

Keep arrays empty and certification blocked in `compatibility_profiles.toml`.

### Decision evidence

- upstream SymPy 1.14 support envelope;
- PyO3/bridge compatibility;
- available CI hardware;
- ecosystem corpus distribution;
- wheel maintenance cost.

### Deadline

Before WS05 profile freeze.

## 28. Open decision D-002 — Big integer substrate

### Options

- own implementation from first code;
- provisional safe Rust crate behind `fsym-bigint`;
- hybrid: provisional bootstrap while an owned implementation matures.

### Evaluation

Correctness, performance, cancellation granularity, deterministic serialization, unsafe/transitive tree, Wasm, and maintenance.

### Default

Hybrid containment is permissible; no final choice is implied by the plan.

### Deadline

Before WS03 API freeze.

## 29. Open decision D-003 — Certified ball arithmetic substrate

### Question

Can the project build a pure-Rust directed-rounding real/complex ball core with adequate rigor and performance, or should a foundational safe crate be provisionally contained?

### Required evidence

Rounding-mode control, transcendental enclosure strategy, architecture parity, proof/verifier interface, and mutation corpus.

### Deadline

Before WS11 certificate claims.

## 30. Open decision D-004 — Exact opaque-node contract

### Question

Which capabilities can a user-defined Python node safely declare for native lowering: commutativity, purity, determinism, derivative, numeric implementation, serialization, code generation?

### Default

Opaque and conservative; unknown capabilities are false/unsupported, not inferred.

### Deadline

Before WS05 native fast paths stabilize.

## 31. Open decision D-005 — First certificate families

### Candidates

Polynomial identity, factorization product/multiplicity, GCD, differentiation, exact linear solve/determinant, certified enclosure.

### Decision criterion

High value, small independent verifier, exercises multiple architecture layers, and supports the hero workload.

### Default

The campaign order in `EVIDENCE_PROOFS_AND_REWRITES.md`.

## 32. Open decision D-006 — Compatibility shell source strategy

### Options

- clean-room/manual shell guided by manifests/oracle;
- source translation with licensing/provenance controls;
- generated repetitive declarations plus manual semantic implementations;
- staged combination.

### Evaluation

License compatibility, maintainability, exact object behavior, diffability across profiles, and risk of embedding upstream runtime assumptions.

### Deadline

Before broad WS05/WS21 expansion.

## 33. Open decision D-007 — Stable digest and canonical encoding

### Question

Which digest algorithm and binary encoding provide long-lived content identity, streaming verification, domain separation, Wasm support, and acceptable performance?

### Requirements

Algorithm agility via schema versions; collision response checks canonical payload; no Python/process hash reuse.

### Deadline

Before persistent public bundles become stable.

## 34. Open decision D-008 — Proof-kernel logic boundary

### Question

How much logic belongs in the tiny kernel versus claim-specific certificate verifiers?

### Default

Minimal equality/congruence/substitution/assumption kernel plus certificate lemmas. Do not build an all-purpose theorem prover before the first certificate campaign.

### Deadline

Before WS06 API freeze.

## 35. Open decision D-009 — Strict compatibility versus hardened behavior packaging

### Question

Should hardened controls be a native namespace, separate distribution, process/service policy, or some combination?

### Constraint

No silent behavior change inside a certified strict profile.

### Deadline

Before first public preview packaging.

## 36. Open decision D-010 — Persistent object granularity

### Question

Persist individual terms/proofs, chunked packfiles, cache segments, or a tiered combination?

### Evaluation

Hot-path isolation, deduplication, privacy leakage, GC, RaptorQ economics, random access, and rebuild cost.

### Deadline

Before WS15 schema freeze.

## 37. Research agenda A — Conditional equality saturation

Develop e-graph structures where equalities carry side-condition contexts without unsound unconditional unions or explosive guard sets.

Questions:

- canonical representation of obligation sets;
- subsumption/dominance under assumptions;
- proof extraction with guarded paths;
- branch-aware analytic rules;
- cost models including proof and condition complexity;
- deterministic bounded parallel saturation.

Success is measured by verified simplification quality and controlled growth, not e-node count.

## 38. Research agenda B — Certificate-rich integration and limits

Explore compact independently checkable evidence for:

- elementary integration decisions;
- rational/algebraic integration;
- branch-aware antiderivatives;
- definite integrals with convergence/singularity partitions;
- asymptotic comparison and remainder bounds;
- creative telescoping;
- transform regions of convergence.

A key objective is separating “candidate differentiates correctly” from complete analytic claims.

## 39. Research agenda C — Modular and sparse exact algebra

Advance pure-Rust, verifier-backed algorithms for:

- sparse multivariate factorization;
- modular Gröbner bases and reconstruction;
- black-box polynomial/matrix methods;
- p-adic lifting;
- interpolation with certified degree/support bounds;
- deterministic unlucky-prime management;
- checkpointable distributed modular work.

## 40. Research agenda D — Certified symbolic-to-numeric compilation

Develop a compilation pipeline where algebraic rewrites, CSE, sparse layouts, branch guards, and target mappings retain proof/certificate provenance.

Targets include:

- FrankenNumPy/FrankenSciPy;
- Rust/Wasm;
- interval/ball evaluators;
- automatic differentiation kernels;
- code-generated exact residual checkers;
- vectorized high-core CPU execution.

## 41. Research agenda E — Anytime-valid adaptive algorithm selection

Study safe online selection under changing symbolic workloads:

- asymmetric loss with catastrophic false-result cost;
- censored outcomes from timeouts/cancellation;
- subgroup calibration by domain/size/architecture;
- shadow strategies and regret proxies;
- e-process drift detection under adaptive logging;
- freeze/revert policies;
- resistance to benchmark/reward hacking.

The selector chooses work, never truth.

## 42. Research agenda F — Incremental proof-aware workspaces

Develop semantic branch/merge for collaborative mathematical work:

- dependency-minimal proof invalidation;
- assumption/rule/context rebase;
- proof reuse across alpha-equivalent or representation-changed terms;
- compact semantic patches;
- counterexample-driven rule quarantine;
- agent division of proof obligations;
- graph-index acceleration without graph authority.

## 43. Research agenda G — Deterministic parallel exact arithmetic

Investigate high-performance algorithms whose outputs and stable artifacts do not depend on scheduling:

- NTT/CRT multiplication and reconstruction;
- deterministic parallel gcd/factorization;
- sparse accumulation order;
- exact matrix reductions;
- stable proof/certificate extraction;
- Apple Silicon and high-core AMD/Intel dispatch.

Performance mode may relax byte-level decision traces but not semantic/evidence correctness.

## 44. Research agenda H — Pure-Rust certified transcendental numerics

Develop rigorous enclosures for elementary and special functions:

- argument reduction with exact constants;
- directed polynomial/rational approximants;
- range/branch reduction;
- complex discs/rectangles;
- asymptotic and recurrence methods;
- proof-producing error bounds;
- vectorized evaluation.

This is a strategic bridge between symbolic exactness and reliable numerical fallback.

## 45. Research agenda I — Compatibility mining

Automate extraction of a dynamic Python library's real contract:

- source/AST/reflection inventories;
- class/metaclass hook detection;
- environment-sensitive behavior probes;
- generated custom subclasses;
- side-effect and cache-state exploration;
- ecosystem import/private-use graphs;
- semantic mismatch minimization;
- profile diffs across upstream releases.

The tooling should be reusable beyond SymPy.

## 46. Research agenda J — Trust-minimized proof infrastructure

Study:

- smaller proof kernels and canonical proof IR;
- certificate composition;
- proof compression with replayable macro steps;
- differential optimized/reference verifiers;
- proof-carrying caches and generated code;
- mutation adequacy measures;
- optional export into external proof assistants without making them runtime dependencies.

## 47. Research agenda K — Artifact-value-aware repair

Develop principled repair policies based on:

- recomputation cost distribution;
- retention horizon and collaboration value;
- correlated failure domains;
- adaptive loss/corruption telemetry;
- source/repair placement;
- privacy and deduplication;
- checkpoint incremental structure;
- proof/replay bundle dependency closure.

RaptorQ is one mechanism within this economic policy, not the policy itself.

## 48. Research agenda L — Agent-native mathematical ergonomics

Measure whether agents perform better with:

- stable structural term/proof references;
- semantic patches versus text rewrites;
- expandable evidence graphs;
- bounded work packets;
- explicit unresolved obligations;
- counterexample bundles;
- plan-only and verify-only APIs;
- deterministic session replay.

Evaluation should include correctness, merge conflict rate, token/transport cost, proof reuse, and reward-hacking resistance.

## 49. Risk review cadence

Risk review occurs:

- before each milestone transition;
- when a blocking discrepancy or unsoundness incident appears;
- before changing a source/profile/dependency pin;
- before enabling a new evidence class or certificate family;
- before public performance/compatibility/security claims;
- after the Certified Jacobian Pipeline campaign;
- before drop-in release certification.

Every review records changed likelihood/severity, evidence, mitigation effectiveness, new triggers, and workstream/claim impacts.

## 50. Architecture-review triggers

The following require more than a local patch:

- a valid SymPy object behavior is unrepresentable by the shell protocol;
- an accepted false mathematical claim;
- a stable-ID collision or noncanonical encoding;
- generator/verifier independence proven illusory;
- cancellation leaves shared state or orphan work;
- persistence/repair changes semantic output;
- hidden upstream fallback enters a product path;
- a profile comparator is shown too weak;
- a security boundary allows cross-tenant formula leakage or code execution;
- dependency/FFI pressure contradicts the memory-safety constitution;
- hero workload cannot compose without bypassing a load-bearing contract.

## 51. Risk-register forbidden shortcuts

- lowering likelihood because mitigation code was started;
- closing risk from one positive benchmark;
- hiding existential risk in a generic backlog issue;
- redefining the claim to avoid recording a failure without claim-registry review;
- treating absence of known counterexamples as proof of soundness;
- accepting a dependency or FFI path because it is wrapped in safe-looking API;
- conflating implementation difficulty with impossibility;
- using e-process confidence as evidence that a risk cannot occur;
- deleting retired risk IDs or incident history;
- declaring research success without a verifier/gate and representative corpus.
