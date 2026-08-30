# Comprehensive plan for FrankenSymPy

**Version:** 1.0 planning baseline  
**Date:** 2026-08-19  
**Status:** public architecture and implementation plan; core implementation landed and green, runtime capabilities are implemented-uncertified and not certified  
**Target:** an independently implemented, memory-safe, agent-native symbolic mathematics system and a certified drop-in replacement for named SymPy profiles

---

## 0. Current reality

This repository contains a substantial, tested implementation workspace together with the architecture, governance, compatibility, proof, runtime, conformance, security, and research plan described here. As of 2026-08-29 the workspace spans 25 crates (~66k lines) with green workspace tests, a Python compatibility shell slice, a live SymPy 1.14.0 differential conformance lane, and landed proof/evidence/runtime substrates.

It does **not** currently contain:

- a certified replacement for SymPy (no compatibility profile is certified);
- certified Python object-model conformance (the shell is a slice with an open differential-gate obligation);
- complete proof-producing algebra/calculus coverage to the planned portfolio breadth;
- demonstrated speedups over the live incumbent;
- live FrankenSQLite, FrankenGraphDB, FrankenNumPy, or FrankenSciPy integrations;
- production RaptorQ checkpoint repair or a closed repair evidence bundle;
- live conformal/e-process monitoring.

Capability status is machine-readable in [`registries/claims.toml`](registries/claims.toml). Claims with landed implementation are marked `implemented_uncertified` with named artifacts; every `validated`/`certified` claim remains open behind its gate bundle.

This explicit separation between target architecture and implementation reality is part of the design, not a disclaimer to be removed later.

---

## 1. Executive thesis

FrankenSymPy should not be “SymPy rewritten in Rust.” That would preserve many of SymPy's structural limitations while breaking much of the Python behavior that makes SymPy useful.

The project should instead combine:

1. **A real Python-compatible object shell** that preserves the selected SymPy profile's dynamic class, metaclass, evaluation, assumptions, printer, pickle, warning, exception, and mutation behavior.
2. **A deterministic native semantic kernel** for exact domains, canonical shared terms, proof-producing transformations, adaptive algorithm portfolios, certified numerics, compilation, and high-performance execution.
3. **A derivation/evidence layer** that separates candidates from accepted results and records proofs, certificates, assumptions, side conditions, decisions, budgets, replay, and provenance.
4. **A structured execution fabric** built on asupersync: nested budgets, cancel-correct regions, deterministic lab execution, resumable continuations, speculative portfolios, remote work, and two-phase verified publication.
5. **An agent-native mathematical workspace** with stable term/proof IDs, semantic patches, proof-aware branches, counterexample bundles, bounded work packets, and deterministic session replay.
6. **Optional durable and distributed substrates** using FrankenSQLite, FrankenGraphDB, RaptorQ, FrankenNumPy, and FrankenSciPy through narrow adapters that never define mathematical truth.
7. **An executable compatibility and claims program** in which every release statement is tied to an immutable profile and same-commit evidence.

The intended leapfrog is therefore not one algorithm. It is a new composition:

> Python object fidelity + native exact algebra + proof-carrying adaptive execution + certified numerics + deterministic resource control + durable agent-native collaboration.

---

## 2. Why a genuine SymPy replacement is unusually difficult

SymPy is both a computer algebra system and a Python object protocol. Its effective contract includes far more than mathematical output.

A replacement must account for:

- subclasses of `Basic`, `Expr`, `Atom`, `Function`, matrices, and many domain-specific classes;
- metaclasses and dynamically created undefined function classes;
- custom `__new__`, classmethod `eval`, `_eval_*`, `_sympy_`, converter, and postprocessor behavior;
- `evaluate=False`, global/thread-local evaluation policy, and deliberately noncanonical forms;
- class-sensitive equality, hashing, comparison, sort keys, `args`, and `func(*args)` reconstruction;
- assumptions implemented by class defaults, instances, inference, local contexts, and custom hooks;
- mutable compatibility objects with alias, copy, indexing, and pickling behavior;
- printers, warnings, exceptions, exact module paths, signatures, descriptors, and serialization;
- ecosystem code that relies on widely used semi-private behavior.

An opaque native `Expr` type cannot preserve this contract. Nor can a compatibility claim be established by porting a subset of the upstream tests or matching printed expressions.

The architecture therefore accepts a hard constraint:

> The Python shell owns Python identity. The Rust kernel owns native mathematical semantics. The boundary between them is explicit and testable.

---

## 3. Source projects and inheritance

The design is grounded in pinned source audits rather than broad family resemblance. Exact revisions and adopt/adapt/reject decisions are recorded in [`docs/SOURCE_PROJECT_AUDIT.md`](docs/SOURCE_PROJECT_AUDIT.md).

### 3.1 asupersync

Inherited:

- region-owned structured concurrency;
- request → drain → finalize cancellation;
- explicit capability contexts and nested budgets;
- deterministic lab schedules, virtual time, and replay;
- speculative racing whose losers drain before return;
- two-phase effect publication;
- RaptorQ mechanisms for selected valuable artifacts;
- anytime-valid monitoring concepts for operational streams.

Adapted:

- budgets gain symbolic dimensions such as term count, coefficient height, proof size, modular primes, branch count, and callback calls;
- runtime outcomes compose with mathematical evidence outcomes;
- adaptive selectors choose work, while independent verifiers decide acceptance.

Rejected:

- detached symbolic background work;
- universal cancellation claims through arbitrary Python callbacks;
- treating an e-process or scheduler confidence as a proof;
- treating RaptorQ decoding as integrity or mathematical validity.

### 3.2 FrankenSQLite

Inherited:

- explicit compatibility/native separation;
- immutable versions and snapshot reasoning;
- durable ledgers, checkpoints, crash recovery, and replay;
- claim fencing between designed, dormant, partial, and live mechanisms;
- evidence-preserving histories and conflict witnesses.

Adapted:

- MVCC becomes workspace/context/rule/proof versioning;
- the database is an optional ledger and persistent cache outside the algebraic hot path;
- semantic branch merge validates universe and proof dependencies.

Rejected:

- row IDs as mathematical identities;
- persistence as a prerequisite for local correctness;
- silent recovery that changes profile, assumptions, rules, or results.

### 3.3 FrankenGraphDB

Inherited:

- a small set of named architectural bets;
- constitutional prohibitions against fake shortcuts;
- authoritative state versus rebuildable derived indexes;
- deterministic plans, certificates, and decision cards;
- branch-per-agent work and semantic merge;
- executable milestone and claim gates.

Adapted:

- graph storage indexes derivations, dependencies, counterexamples, and collaborative work;
- authoritative term/proof objects remain content-addressed outside the graph projection.

Rejected:

- graph reachability as logical entailment;
- a graph server as a requirement for ordinary local symbolic computation.

### 3.4 FrankenNumPy

Inherited:

- full public-surface inventory;
- isolated differential oracle processes;
- machine-readable discrepancy and evidence ledgers;
- strict versus native/hardened operating postures;
- parity before performance;
- format hardening, fuzzing, and adversarial fixtures.

Adapted:

- array-layout compatibility becomes Python object-model/evaluation/assumptions compatibility;
- numerical comparators expand into exact surface, structure, exception, warning, set, branch, and certified-enclosure comparators.

Rejected:

- exported-name reachability as parity;
- hidden upstream fallback in certified artifacts;
- benchmark aggregates containing incompatible cases.

### 3.5 FrankenSciPy

Inherited:

- condition-aware algorithm portfolios;
- explicit state/diagnostic models;
- asymmetric loss functions;
- safe baselines, shadow strategies, and fallback graphs;
- decision cards and calibrated selection.

Adapted:

- the selector's confidence affects launch order, never mathematical evidence;
- every accepted candidate passes a claim-specific verifier;
- exact, certified numeric, compatibility, and heuristic outcomes remain distinct.

Rejected:

- posterior probability as proof;
- heuristic simplifier output presented as an exact transformation;
- speed claims without a live incumbent and semantic admission.

---

## 4. The seven architectural bets

The project is organized around seven load-bearing compositions.

### Bet 1 — Dual-lane compatibility

A profile-compatible Python object shell preserves observable behavior. Eligible built-in regions lower into a native kernel. Arbitrary user extensions remain ordinary Python objects and cross a supervised opaque-node boundary.

### Bet 2 — Three-graph truth model

FrankenSymPy keeps separate:

1. **Surface Object Graph (SOG):** exact class, ordered args, held/evaluation policy, mutable aliases, and reconstruction behavior.
2. **Semantic Term DAG (STD):** immutable typed canonical terms for native algorithms.
3. **Derivation Evidence Graph (DEG):** claims, transformations, side conditions, proof/certificate edges, decisions, and receipts.

### Bet 3 — Domain-explicit exactness

Every semantic operation names its domain, assumptions context, branch policy, coercion decisions, and identity universe. Approximation is explicit and certified when claimed.

### Bet 4 — Proof-carrying portfolios

Polynomial, linear algebra, simplification, integration, solving, and other engines may race multiple strategies. Candidate generation can be complex; acceptance is controlled by smaller independent verifiers.

### Bet 5 — Deterministic resource sovereignty

All controlled work is asupersync-owned, multidimensionally budgeted, cancellation-aware, two-phase published, and replayable under declared modes.

### Bet 6 — Agent-native symbolic state

Agents receive structured mathematical objects and machine-readable obligations rather than strings. Semantic patches and branch merges are verifier-governed.

### Bet 7 — Recoverable computation fabric

Expensive work can be checkpointed, repaired, resumed, distributed, indexed, and replayed while keeping bytes, integrity, authorization, schema validity, and mathematical evidence as separate layers.

The full constitutional rules are in [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md).

---

## 5. Product and packaging model

### 5.1 Native Rust library

The Rust library exposes:

- term/domain/context construction;
- exact and certified numeric towers;
- typed mathematical requests and outcomes;
- proof/certificate verification;
- structured budgets and cancellation;
- deterministic/replay execution;
- persistence/distribution traits;
- protocol and bundle schemas;
- no dependency on Python for native use.

### 5.2 `frankensympy`

A coexistable Python distribution under its own namespace provides:

- native APIs and result/evidence envelopes;
- preview compatibility shell implementations;
- explicit profile inspection;
- plan/verify/budget/replay/persistence controls;
- adapters to the Franken numeric ecosystem;
- no implication that it is already a complete `sympy` replacement.

### 5.3 `frankensympy-dropin`

A separate distribution owns the top-level `sympy` package and intentionally conflicts with upstream SymPy in one environment. It is published as certified only when one immutable compatibility profile passes all release gates.

### 5.4 CLI and protocol service

A robot-first CLI and NDJSON/RPC interface expose the same typed requests, events, results, proofs, workspaces, and bundles as the in-process APIs.

### 5.5 WebAssembly subset

A native subset targets browsers and sandboxed runtimes: terms, selected exact algebra/calculus, proof checking, certified numeric primitives, protocol/bundle handling, and generated evaluators. It excludes CPython and unsupported persistence/threading features.

---

## 6. Immutable compatibility profiles

Compatibility is a versioned empirical contract. The first certification target is provisionally:

```text
sympy-1.14.0-cpython
```

Pinned upstream source:

```text
16fa855354eb7bcabd3fe10993841e03b1382692
```

The profile records:

- upstream source/test digests;
- CPython versions, ABIs, and platform tags;
- optional dependency sets;
- modules, exports, aliases, classes, metaclasses, MROs, descriptors, and signatures;
- construction/evaluation/equality/hash/sort/assumptions policies;
- warnings, exceptions, printers, pickles, mutation, and copy behavior;
- complete test inventory and exclusions;
- comparator registry and discrepancy digest;
- rule, algorithm, verifier, lowering, and schema versions.

Changing any identity-relevant field creates a new profile. A moving-head lane may identify future drift but cannot modify an existing certification theorem.

Detailed contract: [`docs/COMPATIBILITY_CONTRACT.md`](docs/COMPATIBILITY_CONTRACT.md).

---

## 7. Python compatibility shell

### 7.1 Shell responsibilities

The shell owns:

- Python module/class/metaclass identity;
- MRO, slots, descriptors, singleton identity, and introspection;
- constructor/evaluation hooks;
- `args`, `func`, equality, hash, compare, and sort keys;
- assumptions state visible to the profile;
- printers, warnings, exceptions, copy, and pickle;
- mutable compatibility objects;
- arbitrary user subclasses and converters.

### 7.2 Native-backed built-ins

Exact built-in classes may store compact native handles. The fast path is used only when class and hook checks prove that no legal Python override can observe different behavior.

### 7.3 Shell-only implementations

A surface can remain implemented in FrankenSymPy's own Python shell until a safe lowering contract exists. Shell-only is not upstream fallback and does not weaken independence.

### 7.4 Opaque custom objects

Unknown/custom nodes receive descriptors containing exact class identity, ordered children when safely observable, declared capabilities, callback interfaces, and serialization status.

They are not assumed:

- commutative;
- pure;
- deterministic;
- terminating;
- thread-safe;
- serializable;
- mathematically interpretable.

### 7.5 Mutable cells

Mutable matrices and related values remain shell-owned. Native algorithms receive immutable snapshots carrying source cell ID, generation, shape/domain metadata, and digest. Mutating APIs validate generation before write-back.

---

## 8. Surface Object Graph

A surface node records profile-visible construction:

```text
SurfaceNode
├── SurfaceId
├── profile and Python class descriptor
├── ordered arguments
├── evaluation/held policy
├── instance state relevant to behavior
├── assumptions state reference
├── mutability/alias generation
├── optional native handle
├── callback capabilities
└── reconstruction/pickle descriptor
```

Distinct surface nodes may map to one semantic term. For example, a held `Add(x, x, evaluate=False)` and canonical `2*x` may be mathematically equal while differing in `args`, printer behavior, reconstruction, and profile observations.

Surface IDs are workspace identities, not mathematical identities.

---

## 9. Semantic Term DAG

A semantic term contains:

```text
SemanticTerm
├── TermId
├── operator ID and schema
├── ordered child TermIds
├── canonical payload
├── domain ID
├── sort/kind
├── binder representation
└── identity-relevant registry version
```

Properties:

- immutable and hash-consed;
- canonical only under operator/domain-specific rules;
- explicit commutative versus noncommutative representations;
- alpha-invariant binders with surface-name mapping;
- content identity independent of memory, Python hash, persistence, or scheduling;
- bounded arenas with generation-checked process-local handles;
- payload confirmation at trust boundaries rather than digest-only equality.

The semantic DAG is optimized for algorithms. It is not a substitute for arbitrary Python class identity.

---

## 10. Derivation Evidence Graph

A derivation edge records:

```text
DerivationEdge
├── typed claim
├── source and target term IDs
├── context/domain/branch policy
├── rule/algorithm/verifier versions
├── proof or certificate reference
├── evidence class
├── side-condition obligations
├── execution/decision/verification receipts
└── parent derivations
```

Only verifier-accepted edges belong to the trusted derivation subgraph. Rejected candidates and search traces may be retained for diagnosis and counterexamples but cannot become proof parents.

The graph supports:

- expandable explanations;
- dependency invalidation;
- proof replay;
- collaborative branches;
- counterexample attachment;
- compact certificate extraction;
- optional FrankenGraphDB indexing.

Full representation design: [`docs/OBJECT_MODEL_AND_IR.md`](docs/OBJECT_MODEL_AND_IR.md).

---

## 11. Stable identity model

Distinct typed IDs include:

- `SurfaceId`;
- `TermId`;
- `DomainId`;
- `AssumptionsContextId`;
- `RuleRegistryId`;
- `AlgorithmRegistryId`;
- `VerifierRegistryId`;
- `ClaimId`;
- `DerivationId`;
- `ReceiptId`;
- `CheckpointId`;
- `BundleId`;
- `WorkspaceId` and `BranchId`.

`TermId` preimages include the canonical operator, payload, ordered children, binders, sort, domain-relevant data, and schema domain. They exclude Python/process hashes, interner handles, rows, pointers, provenance, time, and planner statistics.

Python `__hash__` remains profile-correct and process-local where upstream behavior requires it.

---

## 12. Lowering and lifting

### 12.1 Lowering

Lowering:

1. freezes profile, evaluation policy, assumptions context, domain/rule universe;
2. snapshots mutable values;
3. resolves exact Python classes and registered lowering contracts;
4. checks overridden hooks and unsafe assumptions;
5. recursively lowers in profile-preserving order;
6. builds native operators or explicit opaque nodes;
7. validates domains/coercions;
8. records every normalization, loss, and opaque boundary.

Outcomes include fully lowered, partially lowered, shell-only, refused, cancelled, and resource-exhausted.

### 12.2 Lifting

Lifting chooses among:

- reusing a valid existing shell object;
- constructing a profile-canonical built-in;
- reconstructing a requested held form;
- returning an explicit native-only wrapper;
- refusing a lossless profile representation.

It accounts for constructor evaluation, class/singleton identity, assumptions, branches, printers, warnings, exceptions, and pickle paths.

---

## 13. Assumptions system

The internal query result is four-way:

```text
EntailedTrue(proof)
EntailedFalse(proof)
Unknown(reason)
Contradictory(witness)
```

The compatibility shell maps these to profile-visible behavior such as `True`, `False`, `None`, or an exception.

Contexts are immutable and content-addressed. Adding/retracting facts creates a child context. Facts retain provenance: definitional, kernel theorem, user assertion, custom Python hook, imported proof, or heuristic observation.

Inference tiers:

1. definitional domain facts;
2. terminating Horn-style predicate closure;
3. operator-specific structural handlers;
4. solver-backed obligations;
5. supervised Python hooks.

Contradiction policies include rejection, paraconsistent retention, and profile behavior. Classical explosion is not the default for inconsistent imported/user contexts.

---

## 14. Domain and coercion system

Core exact domains include:

- arbitrary-precision integers and rationals;
- finite rings/fields and extensions;
- Gaussian and algebraic extensions;
- exact real algebraic numbers;
- polynomial, Laurent polynomial, power-series, and fraction-field domains;
- quotient rings and ideals;
- matrix rings/modules;
- noncommutative algebras where supported;
- Boolean/logic theories;
- generic expression domains as explicit escape hatches.

Certified analytic domains include real/complex intervals or balls and later affine/Taylor models where justified.

Every coercion edge states:

- total or partial;
- injective or lossy;
- exact or approximate;
- required assumptions;
- inverse/reconstruction support;
- cost and evidence;
- profile-visible behavior.

Ambiguous coercions are resolved deterministically or refused, never selected by incidental plugin registration order.

---

## 15. Exact and certified numeric tower

The target tower distinguishes:

```text
Integer
Rational
FiniteFieldElement
AlgebraicNumber
RealAlgebraic
ExactConstant
ProfileFloat
RealBall
ComplexBall
Infinity variants
NaN/indeterminate forms
SymbolicNumber
```

### 15.1 Exact arithmetic

The core requires a pure-Rust, memory-safe arbitrary-precision substrate with canonical serialization, cancellation safe points, scalar references, and portfolios for schoolbook/Karatsuba/Toom/NTT-scale operations.

The exact initial substrate—owned from the beginning or a provisional safe Rust crate behind a removable façade—is an explicit open decision.

### 15.2 Certified numerics

Certified values carry directed-rounding enclosures, target term/context, precision policy, branch/singularity analysis, error/termination certificate, and verification receipt.

A narrow interval is not certified merely because a generator reports it. Exact recognition through integer relations or rational reconstruction remains a candidate until independently checked.

Detailed design: [`docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md`](docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md).

---

## 16. Result and evidence model

Native APIs return a structured outcome:

```text
Accepted(value, claim, evidence, verification receipt, execution receipt)
Conditional(value, unresolved obligations)
HeuristicCandidate(value, diagnostics)
Inconclusive(explored methods, continuation)
Refused(reason)
Cancelled
TimedOut
ResourceExhausted
Unsupported
InternalFault
```

Evidence classes include:

- **KernelProved**;
- **CertificateVerified**;
- **ExactCrossChecked**;
- **CertifiedNumeric**;
- **OracleConformant**;
- **UserAsserted**;
- **HeuristicCandidate**.

These are not one scalar confidence ladder. A compatibility observation and a mathematical proof answer different questions.

Prohibited promotions are machine-readable in [`registries/evidence_classes.toml`](registries/evidence_classes.toml).

---

## 17. Proof kernel and certificate architecture

The initial proof language includes:

- reflexivity, symmetry, transitivity;
- congruence;
- assumptions;
- definitional reduction;
- registered conditional rewrite with side-condition proofs;
- implication and equality substitution;
- capture-avoiding binder renaming;
- domain embeddings;
- verified certificate lemmas;
- certified numeric separation.

Certificate families include:

- polynomial identity;
- GCD and divisibility;
- factorization with explicitly scoped irreducibility/completeness;
- Gröbner basis and ideal membership;
- exact linear solve/decomposition/determinant/rank/nullspace;
- root isolation and completeness;
- differentiation;
- integration/limit/series claims with branch/convergence conditions;
- SAT/logic proof traces;
- certified numeric enclosures.

Generators and verifiers are dependency-separated. High-value optimized verifiers retain simple reference implementations. Mutation tests deliberately weaken product checks, S-pair coverage, residual equality, multiplicity, side conditions, directed rounding, registry binding, and stored verification metadata.

Detailed design: [`docs/EVIDENCE_PROOFS_AND_REWRITES.md`](docs/EVIDENCE_PROOFS_AND_REWRITES.md).

---

## 18. Rewrite and simplification engine

Rewrite rules are registry objects containing patterns, sorts, match theory, side conditions, domains, branch policy, direction, termination/cost metadata, compatibility visibility, and proof constructors.

Two execution forms coexist:

### Deterministic local rewriting

Used for profile-critical construction and small canonical transforms. Rule order is profile/version controlled, bounded, and proof-producing.

### Goal-directed bounded search

Native simplification accepts a cost vector covering expression/DAG size, target operation count, coefficient height, branch complexity, numerical stability, code-generation cost, proof size, and compatibility surface distance.

Bounded local e-graphs are search devices, not global expression storage. Conditional rules create guarded relations, and every e-class union retains a justification that can be extracted and verified.

---

## 19. Proof-carrying algorithm portfolios

Every major operation is eligible for a portfolio with:

- instance evidence vector;
- state/regime model;
- eligible strategies;
- asymmetric loss policy;
- safe baseline;
- decision card;
- protected verifier budget;
- two-phase candidate publication;
- fallback graph;
- deterministic/replay policy;
- monitoring and learning rules.

The selector may consider domain, degree, sparsity, coefficient height, symmetry, matrix structure, branch obligations, proof cost, available cores, memory, and prior verified outcomes.

False accepted exact results carry catastrophic loss. Inconclusive or slow exact results are materially different and must not be optimized as if all failures were equivalent.

Full domain-by-domain portfolio program: [`docs/ALGORITHM_PORTFOLIOS.md`](docs/ALGORITHM_PORTFOLIOS.md).

---

## 20. Algebra program

### 20.1 Polynomial representations

Adaptive representations include:

- dense univariate;
- sparse distributed multivariate;
- recursive towers;
- modular images;
- evaluation/interpolation black boxes;
- straight-line programs;
- truncated series.

Conversions emit receipts and invariant checks.

### 20.2 GCD and resultants

Strategies include Euclidean/subresultant, modular with unlucky-prime handling, evaluation/interpolation, sparse/black-box, and heuristic candidate generation followed by exact verification.

### 20.3 Factorization

Target strategies include content/square-free decomposition, finite-field factorization, Hensel lifting, deterministic/LLL-style recombination, algebraic extensions, sparse multivariate evaluation/lifting, and absolute factorization research.

The requested claim explicitly states decomposition versus irreducibility, multiplicity, domain, normalization, and completeness.

### 20.4 Gröbner and ideals

Portfolio targets include Buchberger, F4, signatures/F5, modular reconstruction, FGLM, elimination, saturation, regular-chain-related methods, and incremental workspace updates.

Certificates establish ideal membership and Gröbner criteria under the exact monomial order/domain.

### 20.5 Algebraic numbers

The program includes defining polynomials, embeddings, isolating regions, exact arithmetic, equality/order certificates, minimal polynomials, primitive-element/tower management, and root objects.

---

## 21. Exact linear algebra program

Dense strategies:

- fraction-free Bareiss;
- denominator-cleared decompositions;
- modular/CRT and p-adic solve/determinant/rank;
- characteristic/minimal polynomial methods;
- structured block methods.

Sparse strategies:

- fill-aware fraction-free elimination;
- sparse modular elimination;
- block Wiedemann/Lanczos candidates with exact verification;
- graph/order-informed pivot planning;
- structured Toeplitz/Hankel/banded paths.

Symbolic parameter strategies use domain matrices, polynomial/fraction-field choices, interpolation, and piecewise singular branches.

Certificates distinguish existence, uniqueness, rank, completeness, invertibility, decomposition, and determinant claims.

---

## 22. Calculus and analysis program

### Differentiation

Structural rules produce proof terms. High-order/multivariate work uses shared-DAG dynamic programming, sparse Jacobian/Hessian analysis, tensor/matrix calculus, and compiled AD. Custom derivative hooks remain provenance-marked.

### Integration

Portfolios include rule systems, substitutions/parts, rational integration, algebraic-function methods, Risch-style components, hypergeometric/Meijer transforms, residues/contours, creative telescoping, ODE-based recognition, heuristic search, and certified quadrature.

Claims distinguish antiderivative, definite integral, principal value, conditional result, certified enclosure, proved non-elementarity under a decision procedure, and “not found.”

### Limits and series

Methods include continuity/substitution, dominant-term algebra, comparison classes, formal power/Laurent/Puiseux/logarithmic series, recurrence methods, sector/direction analysis, remainder bounds, certified neighborhoods, and later transseries research.

Formal algebra and analytic convergence are separate claims.

### Sums, products, and transforms

Telescoping/recurrence/creative-telescoping evidence includes boundary and convergence conditions. Fourier/Laplace/Mellin/Hankel/Z transforms carry regions, conventions, branch data, and inverse-transform obligations.

---

## 23. Solver program

### Algebraic equations

- factor/root isolation;
- resultants and Gröbner elimination;
- rational univariate representations;
- triangular/regular-chain methods;
- CAD and quantifier elimination;
- parameter stratification.

### Transcendental equations

- branch-aware function inversion;
- Lambert W and special forms;
- monotonicity/convexity partitions;
- interval Newton/Krawczyk isolation;
- periodic-family representation.

### Inequalities and sets

- exact sign decomposition;
- CAD/virtual substitution;
- interval/monotonicity certification;
- structural/lazy set algebra;
- solver-backed membership/subset/emptiness.

### Logic

- Boolean canonical forms;
- BDD/AIG strategies;
- SAT with checkable models/UNSAT traces;
- supported arithmetic theories;
- profile fuzzy/three-valued logic.

### ODE and PDE

Solutions are checked by substitution and initial/boundary conditions. A verified solution, general solution, complete family, and singular solution are distinct claims. PDE modules avoid implying completeness from one residual-zero family.

All solvers return explicit completeness status in native mode.

---

## 24. Structured mathematics

### Tensor/index calculus

Typed index spaces, variance, symmetries, stabilizer chains, canonicalization under permutation groups, contractions, and sparse structures replace naive factorial search.

### Geometry

Exact predicates and algebraic intersections are primary, with certified numeric fallback and explicit degeneracy classes.

### Statistics

Random variables, events, distributions, expectations, transforms, moments, conditioning, and stochastic processes distinguish exact, formal, and certified numeric results.

### Units and dimensions

Dimension vectors, affine/log units, conversion graphs, unit systems, and constants with edition/source provenance become typed semantic metadata.

### Physics/control modules

Mechanics, quantum, vector, continuum, optics, control, and related modules compose the same term/domain/evidence system rather than creating isolated mini-kernels.

---

## 25. Symbolic-to-numeric compilation

FrankenSymPy compiles verified symbolic state into numeric programs for Rust, Wasm, FrankenNumPy, FrankenSciPy, and emitted source targets.

The pipeline includes:

- DAG-aware common-subexpression elimination;
- target-aware exact rewrites;
- Horner/Estrin/Paterson–Stockmeyer polynomial evaluation;
- sparse Jacobian/Hessian layout and coloring;
- branch-safe piecewise lowering;
- domain guards;
- SIMD/parallel loop plans;
- exact or certified reference evaluators;
- generated test vectors and provenance.

Generated artifacts carry source term/context IDs, transformation proofs/receipts, target ABI/floating policy, guard definitions, and content digests.

---

## 26. Structured runtime and budgets

Every significant operation receives a capability context containing cancellation, deadlines, budget tree, region, determinism policy, universe IDs, security capabilities, and optional persistence/remote traits.

Budget dimensions include:

- wall and virtual time;
- CPU/fuel and allocation/live memory;
- term and surface nodes;
- expression/rewrite/e-graph growth;
- coefficient height, monomials, degree, primes;
- matrix dimensions/fill/field operations;
- branch and assumptions solver work;
- proof/certificate bytes;
- Python callbacks;
- persistence, remote, repair, output, and printer resources.

Generators cannot consume resources reserved for verification. Fallback does not reset accounting.

Cancellation follows request → drain → finalize. Shared results, caches, and checkpoints publish only through prepare/verify/commit boundaries.

Detailed runtime contract: [`docs/RUNTIME_BUDGETS_AND_DETERMINISM.md`](docs/RUNTIME_BUDGETS_AND_DETERMINISM.md).

---

## 27. Determinism and replay

Modes:

- **strict deterministic:** fixed probes, strategies, seeds, tie breaks, result form, and scoped bytes;
- **replay deterministic:** adaptive production decisions recorded and reproducible;
- **latency adaptive:** live telemetry changes execution, while accepted values remain verifier-governed;
- **compatibility profile:** reproduces upstream-visible ordering/form policy.

Sources of accidental nondeterminism—hash maps, work stealing, random stream consumption, reductions, remote arrival, filesystem enumeration, plugin order—are controlled explicitly.

The lab runtime injects schedules, cancellation, budget failures, torn writes, worker faults, callback delays, and RaptorQ symbol loss against the same runtime-facing interfaces as production.

Replay bundles contain canonical requests/objects, profile/context/registries, policies, seeds/decision traces, proof/certificate/checkpoint artifacts, environment/build fingerprint, and expected terminal digest.

---

## 28. Agent-native protocol

NDJSON/RPC envelopes expose:

- versioned request IDs and parent traces;
- typed term/context/profile/registry references;
- operation, claim, evidence, completeness, branch, budget, and execution policy;
- streaming accepted/plan/candidate/verification/checkpoint/cancellation/terminal events;
- structured accepted, conditional, heuristic, inconclusive, refused, and operational outcomes;
- proof and receipt expansion;
- object chunking and optional repair sidecars;
- exact error codes and bounded diagnostics.

Text parsing produces a surface graph and lowering receipt; strings never become authoritative IDs.

Agents can fork branches, submit semantic patches, attach proof/certificate candidates and counterexamples, request independent verification, and merge only accepted edges under compatible universes.

Detailed protocol: [`docs/AGENT_NATIVE_PROTOCOL.md`](docs/AGENT_NATIVE_PROTOCOL.md).

---

## 29. Persistence and workspace history

Persistence modes:

- ephemeral;
- local durable;
- collaborative workspace;
- distributed execution.

An optional FrankenSQLite-backed ledger stores immutable universe manifests, term/proof/certificate/checkpoint objects, branch events, decisions, receipts, terminal outcomes, cache metadata, worker leases, repair records, and release evidence.

It does not define term equality or proof validity.

Workspace computations read immutable snapshots. New assumptions/rules/derivations create later versions. Merge validates common ancestry, universe compatibility, proof edges, contradictions, mutable snapshots, and capability policy.

Persistent cache entries carry full semantic keys and evidence class. Candidate and verified namespaces remain separate. Reads validate canonical payload, digest, universe, evidence minimum, and verifier policy.

---

## 30. RaptorQ artifact repair

RaptorQ is applied selectively to:

- expensive checkpoints;
- proof/replay archives;
- distributed work packets/results;
- costly verified cache segments;
- release/conformance/benchmark evidence packs;
- minimized fuzz/counterexample corpora.

It is normally excluded from tiny terms, disposable indexes, and cheap rebuildable caches.

Trust sequence:

1. RaptorQ reconstructs candidate bytes.
2. Canonical digest validates expected content.
3. Authorization/signature validates origin where needed.
4. Schema and invariants validate structure.
5. Mathematical verifier validates evidence.

Adaptive redundancy may use artifact value, recomputation cost, retention, storage/transfer cost, observed loss rates, and failure-domain placement. E-process alarms may change future policy; they do not guarantee any artifact or validate its contents.

Detailed design: [`docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md`](docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md).

---

## 31. Remote workers and graph indexing

Remote workers receive bounded content-addressed work packets with exact subgoals, universe IDs, allowed strategy, budget, deterministic seed lease, certificate schema, and capabilities.

Responses are candidates. The local coordinator bounds, canonicalizes, packet-binds, deduplicates, and verifies them. Workers cannot publish branch heads or verified caches.

FrankenGraphDB may index term/operator dependencies, derivations, proof use, rule firings, counterexamples, branches, compatibility discrepancies, generated-code provenance, and agent work. It remains a rebuildable projection; reachability is not proof.

---

## 32. Conformal e-process and anytime-valid monitoring

Monitoring targets streams such as:

- compatibility mismatches;
- verifier rejection and mutation survival;
- selector regret proxies;
- subgroup performance drift;
- proof/runtime/size anomalies;
- numerical enclosure failures;
- cache corruption;
- worker defect rates;
- RaptorQ loss/scrub outcomes.

Every monitor records its assumptions, filtration, subgroup and reset policy, action thresholds, and policy version.

Actions may include pause, quarantine, rollback, increased shadow verification, or investigation. Monitor output never upgrades or rejects an individual mathematical claim.

---

## 33. Conformance laboratory

The laboratory uses isolated processes for:

- immutable upstream SymPy oracle;
- FrankenSymPy compatibility implementation;
- native/reference implementation;
- mathematical verifier;
- comparator/minimizer/ledger;
- artifact publisher.

Conformance covers:

- source/reflection API inventory;
- upstream tests;
- generated type/domain/assumptions-aware expressions;
- arbitrary custom subclasses and hooks;
- held/evaluated forms;
- equality/hash/sort under multiple seeds;
- warnings, exceptions, printers, pickles;
- mutable alias/copy behavior;
- metamorphic properties;
- proof/verifier mutation;
- fuzzing;
- concurrency/cancellation schedules;
- crash/corruption/repair;
- ecosystem packages and notebooks.

Comparators are chosen before execution and profile-versioned. A comparator cannot be weakened merely to make a mismatch disappear.

Full program: [`docs/CONFORMANCE_AND_BENCHMARKING.md`](docs/CONFORMANCE_AND_BENCHMARKING.md).

---

## 34. Parity-gated performance program

For every benchmark case:

1. construct the same canonical/surface input;
2. run upstream/live incumbent and FrankenSymPy candidate under matched policy;
3. compare profile and mathematical observations;
4. invoke required verifiers;
5. ledger failing cases;
6. time only admitted cases;
7. publish paired raw data and aggregates.

Reports include commits, profile, toolchains, hardware, cache/durability modes, threads/workers, evidence requirement, memory, tails, proof/certificate size, outcome mix, and startup/amortization.

Target opportunities include:

- shared-DAG batch workloads;
- large sparse/exact polynomial algebra;
- modular factorization/Gröbner/linear algebra;
- sparse Jacobian/Hessian generation;
- repeated compiled numeric evaluation;
- agent portfolios and proof/cache reuse;
- parallel high-core CPU workloads.

No universal speedup or numeric target is claimed before evidence.

---

## 35. Security and resource governance

Threats include hostile expressions, deep recursion, expansion bombs, AC/e-graph explosion, coefficient swell, malicious proofs/certificates, pickle/code execution, callbacks, cache poisoning, storage corruption, remote workers, cross-tenant ID leakage, and generated code.

Postures:

- strict compatibility;
- native safe;
- hardened multi-tenant.

Core controls:

- safe Rust ordinary crates;
- no C/C++ CAS or arbitrary-precision FFI;
- bounded parsers and canonical formats;
- explicit unsafe pickle capability;
- supervised Python/plugin boundary;
- multidimensional budgets and admission control;
- least-privilege remote/object capabilities;
- candidate/verified namespace separation;
- local proof verification;
- privacy-scoped object stores and metrics;
- dependency/SBOM/provenance gates.

A memory-safe native-core claim does not cover CPython or arbitrary third-party extensions.

Detailed threat model: [`docs/SECURITY_AND_RESOURCE_GOVERNANCE.md`](docs/SECURITY_AND_RESOURCE_GOVERNANCE.md).

---

## 36. Crate and package architecture

Layering:

```text
L7 Product packaging, CLI, Python distributions
L6 Protocol, Python bridge, Wasm, codegen targets
L5 Persistence, distribution, graph index, repair adapters
L4 Planning, portfolios, workspaces, compilation, services
L3 Symbolic algorithm generators
L2 Terms, domains, assumptions, claims, proof kernel, verifiers
L1 Exact arithmetic, canonical encoding, deterministic collections
L0 IDs, budgets, outcomes, schemas, capabilities
```

Key properties:

- verifier crates cannot depend on optimizing generator crates;
- core semantic crates cannot depend on Python, persistence, graph, or network layers;
- asupersync is exposed through a narrow `fsym-cx` contract;
- the CPython bridge is contained and uses audited safe Rust scaffolding rather than hand-written C API code;
- profile-specific Python classes remain Python source, not one extension type;
- optional integrations cannot alter term/proof semantics;
- generated code derives deterministically from reviewed registries;
- dependency/layering/unsafe rules are CI-enforced.

Detailed crate proposal: [`docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md`](docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md).

---

## 37. Dependency posture

Preferred universe:

- Rust standard library;
- asupersync;
- narrow Franken-suite adapters;
- a very small number of foundational crates admitted by written review.

Every new dependency records need, transitive tree, features, unsafe/FFI/build-script/network behavior, determinism, serialization, platform/Wasm effect, maintenance, license, containment, and removal strategy.

Prohibited:

- a second async runtime;
- C/C++ CAS/big-number FFI;
- hidden upstream SymPy execution;
- opaque external solver binaries presented as verified exact engines;
- runtime code loaders;
- repository-wide unsafe allowances.

Open foundational decisions—big integer, ball arithmetic, CPython matrix, digest, opaque-node capabilities—are tracked in the risk/research document.

---

## 38. Workstream graph

The implementation program contains 24 workstreams in [`registries/workstreams.toml`](registries/workstreams.toml):

- WS00 governance/registries/claims;
- WS01 conformance lab;
- WS02 foundation IDs/schemas/budgets/`Cx`;
- WS03 exact arithmetic;
- WS04 terms/domains/assumptions/bindings;
- WS05 Python shell;
- WS06 proof kernel/evidence;
- WS07 rewriting/simplification;
- WS08 polynomial arithmetic;
- WS09 GCD/factorization;
- WS10 exact linear algebra;
- WS11 certified numerics/algebraic numbers;
- WS12 differentiation/compilation;
- WS13 portfolios/runtime;
- WS14 agent protocol/workspaces;
- WS15 persistence/checkpoints/repair;
- WS16 remote work/graph indexing;
- WS17 Gröbner/ideal algebra;
- WS18 integration/limits/series/transforms;
- WS19 solvers/sets/logic/ODE/PDE;
- WS20 structured domains;
- WS21 compatibility/ecosystem closure;
- WS22 performance optimization;
- WS23 packaging/release certification.

The graph is detailed in [`docs/WORKSTREAM_GRAPH.md`](docs/WORKSTREAM_GRAPH.md). Every workstream has dependencies, objective artifacts, gates, and forbidden shortcuts. All currently remain `planned`.

---

## 39. Milestones

### M0 — Planning substrate and claim discipline

Registries, constitutional rules, source pins, claim linter, and work graph.

### M1 — Native semantic nucleus

Typed identities, exact arithmetic, terms, domains, contexts, binders, and initial proof kernel work without Python.

### M2 — Python object-model vertical slice

Initial SymPy profile surface, custom subclasses/hooks, held forms, printers, pickles, and native lowering/lifting pass differential gates.

### M3 — First proof-carrying algebra portfolio

Adaptive polynomial representations, GCD/factorization, independent certificates, cancellation, and parity-gated benchmarks.

### M4 — Certified Jacobian hero pipeline

Proof-producing differentiation, exact matrix subset, certified numerics, and verified numeric compilation compose with M2/M3.

### M5 — Durable and agent-native fabric

Semantic branches, protocol/replay, checkpoint repair/resume, remote candidates, and rebuildable graph index.

### M6 — SymPy 1.14 compatibility expansion

A broad profile checkpoint with a shrinking public discrepancy ledger; not yet certification.

### M7 — Breadth, optimization, and ecosystem closure

Gröbner, calculus, solvers, structured domains, ecosystem corpus, and evidence-backed performance closure.

### M8 — Certified drop-in 1.0

At least one immutable profile and supported platform matrix passes the entire release bundle.

Milestones are gates, not dates or completion percentages.

---

## 40. First implementation campaign: Certified Jacobian Pipeline

The first campaign intentionally forces the architecture to compose.

A user constructs a parameterized nonlinear residual system containing exact polynomial blocks, transcendental built-ins, a deliberately held expression, a custom `Function` subclass, assumptions, and a mutable matrix snapshot.

The system must:

1. preserve profile-visible Python behavior;
2. lower eligible regions and retain opaque/custom nodes;
3. compute a sparse symbolic Jacobian with proof/provenance;
4. factor exact polynomial subexpressions using a two-strategy verified portfolio;
5. compile residual/Jacobian evaluators for FrankenNumPy/FrankenSciPy;
6. produce certified numeric enclosures;
7. checkpoint and cancel with zero orphan work;
8. repair damaged checkpoint bytes with RaptorQ, validate digest/schema/dependencies, and resume;
9. export/replay the same terminal semantic/evidence digest;
10. fork an agent branch, verify a semantic optimization patch, and merge it;
11. reject an invalid remote candidate without cache/branch pollution;
12. benchmark only cases that pass compatibility and mathematical admission.

The full 11-stage campaign and target command contracts are in [`docs/FIRST_IMPLEMENTATION_CAMPAIGN.md`](docs/FIRST_IMPLEMENTATION_CAMPAIGN.md).

Passing the campaign proves one deep architecture slice. It does not prove complete SymPy compatibility.

---

## 41. Beads and agent execution gate

A workstream becomes executable agent work only when decomposed into bounded tasks containing:

- exact objective and non-goals;
- dependency/workstream IDs;
- owned crates/files/registries;
- immutable input universe;
- implementation deliverable;
- independent gate owner;
- acceptance commands;
- unit/property/differential/metamorphic/adversarial tests;
- benchmark obligations and live incumbent;
- discrepancy/claim effects;
- cancellation/resource/failure behavior;
- forbidden shortcuts;
- closure artifacts.

“Implement integration,” “finish compatibility,” or “make factorization fast” are invalid tasks.

Structural graph changes are single-writer and acyclic. Natural-language completion assertions have no authority.

---

## 42. Risk register

Existential risks include:

- an incomplete Python object-model boundary;
- compatibility scope exceeding implementation capacity;
- pure-Rust arithmetic/certified numeric inadequacy;
- verifier/generator non-independence;
- evidence inflation;
- assumptions/branch unsoundness;
- surface fidelity destroyed by canonicalization;
- generated symbolic-to-numeric errors;
- expression/proof denial of service;
- remote/privacy failures.

Critical/high risks include cancellation at Python boundaries, portfolio overhead, determinism/performance tension, persistence contamination, unearned RaptorQ complexity, statistical misuse of e-processes, mutable-object invalidation, serialization/pickle traps, benchmark reward hacking, Franken-suite coupling, dependency minimalism, build fragmentation, documentation drift, and agent-swarm inconsistency.

Each risk has triggers, mitigation, closure evidence, and residual uncertainty in [`docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md`](docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md).

---

## 43. Open decisions

The first architecture review must resolve, with evidence:

1. exact initial CPython/platform profile matrix;
2. owned versus provisional big integer substrate;
3. certified real/complex ball substrate;
4. opaque Python-node capability contract;
5. first certificate families and trusted-kernel boundary;
6. compatibility shell source/generation strategy;
7. canonical encoding and digest algorithm;
8. strict versus hardened packaging;
9. persistent object granularity;
10. which hero root-box claim can honestly be certified in the initial campaign.

Defaults are conservative and certification-blocking rather than guessed.

---

## 44. Research agenda

Strategic research areas:

- conditional equality saturation with proof-carrying guards;
- compact certificates for integration, limits, sums, and transforms;
- sparse/modular exact algebra and distributed reconstruction;
- proof-preserving symbolic-to-numeric compilation;
- anytime-valid adaptive algorithm selection;
- incremental proof-aware collaborative workspaces;
- deterministic parallel exact arithmetic;
- pure-Rust certified transcendental numerics;
- automated compatibility mining for dynamic Python libraries;
- trust-minimized proof infrastructure;
- artifact-value-aware repair policy;
- agent-native mathematical ergonomics and evaluation.

Research success requires a representative corpus and verifier/gate, not an impressive isolated result.

---

## 45. Claims governance

The claims registry distinguishes:

- `documented`;
- `planned`;
- `implemented_uncertified`;
- `validated`;
- `certified`;
- `blocked`;
- `retired`.

Rules:

- present-tense capability claims require a non-planned status;
- implementation requires a live artifact;
- validation requires a gate bundle;
- certification requires same-commit immutable release evidence;
- performance requires live incumbent and semantic admission;
- mathematics requires typed claim and verifier;
- compatibility requires immutable profile;
- repair requires decode, digest, schema/dependency, and semantic checks;
- monitoring cannot grant mathematical evidence.

The only current present-tense project claim is that the source-pinned public planning package exists.

---

## 46. Definition of 1.0

FrankenSymPy 1.0 requires at least one certified drop-in profile and supported platform matrix satisfying all of the following on one release commit:

### Compatibility

- complete public/semi-private inventory for the profile;
- applicable upstream suite;
- generated differential and metamorphic corpora;
- custom subclass/metaclass/hook coverage;
- held/mutable/printer/pickle coverage;
- ecosystem corpus;
- no blocking discrepancy;
- no hidden upstream runtime fallback.

### Mathematical trust

- typed native outcomes and non-inflatable evidence;
- small proof kernel and live certificate verifiers for claimed families;
- mutation/adversarial suites;
- exact/certified reference lanes;
- side-condition/domain/branch correctness.

### Runtime

- asupersync-only owned work;
- multidimensional budgets;
- two-phase verified publication;
- cancellation/drain gates;
- deterministic/replay contracts;
- bounded protocols/decoders.

### Durability/distribution, if shipped

- crash consistency;
- checkpoint resume;
- RaptorQ loss/corruption recovery with digest/schema/proof separation;
- local verification of remote candidates;
- graph-index rebuildability.

### Performance

- every measured case semantically admitted;
- live upstream/reference baselines;
- named workload wins with raw evidence;
- protected memory/tail/cancellation/proof metrics;
- no benchmark reward hacking.

### Security/release

- ordinary native crates safe Rust;
- dependency/SBOM/provenance audit;
- platform wheels/Wasm subset as claimed;
- profile/package conflict/coexistence tests;
- same-commit claims and evidence bundle;
- incident/invalidation/rollback procedures.

1.0 does not imply automatic support for future SymPy releases or universal performance superiority. Each profile and performance statement remains a separate executable claim.

---

## 47. Forbidden shortcuts

The following block release and architecture acceptance:

- hidden upstream SymPy fallback;
- one opaque Rust expression type presented as drop-in compatible;
- strings/printers as semantic identity;
- held/custom surface behavior erased by canonicalization;
- unknown assumptions coerced to truth values;
- branch/side-condition rules applied unconditionally;
- heuristic, posterior, e-process, oracle, sampled numeric, worker, or majority evidence presented as proof;
- generator self-verification;
- verifier dependency on generator;
- first-completed candidate publication;
- detached/orphan work;
- fallback budget reset;
- process memory dumps as checkpoints;
- database rows, graph reachability, cache flags, or RaptorQ decode as truth;
- repaired artifacts used before digest/schema/dependency/evidence checks;
- remote workers publishing verified state;
- arbitrary pickle/code execution through normal protocols;
- C/C++ CAS/big-number FFI hidden by wrappers;
- incompatible cases in benchmark aggregates;
- comparator/golden/test weakening to land features;
- API stubs counted as parity;
- plan language presented as runtime fact;
- milestone closure by prose, percentage, or commit count;
- broad API expansion before the Certified Jacobian Pipeline validates the architecture.

---

## 48. Document and registry map

### Core contracts

- [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md)
- [`docs/SOURCE_PROJECT_AUDIT.md`](docs/SOURCE_PROJECT_AUDIT.md)
- [`docs/COMPATIBILITY_CONTRACT.md`](docs/COMPATIBILITY_CONTRACT.md)
- [`docs/OBJECT_MODEL_AND_IR.md`](docs/OBJECT_MODEL_AND_IR.md)
- [`docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md`](docs/ASSUMPTIONS_DOMAINS_AND_NUMERIC_TOWER.md)
- [`docs/EVIDENCE_PROOFS_AND_REWRITES.md`](docs/EVIDENCE_PROOFS_AND_REWRITES.md)
- [`docs/ALGORITHM_PORTFOLIOS.md`](docs/ALGORITHM_PORTFOLIOS.md)
- [`docs/RUNTIME_BUDGETS_AND_DETERMINISM.md`](docs/RUNTIME_BUDGETS_AND_DETERMINISM.md)
- [`docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md`](docs/PERSISTENCE_DISTRIBUTION_AND_REPAIR.md)
- [`docs/AGENT_NATIVE_PROTOCOL.md`](docs/AGENT_NATIVE_PROTOCOL.md)
- [`docs/CONFORMANCE_AND_BENCHMARKING.md`](docs/CONFORMANCE_AND_BENCHMARKING.md)
- [`docs/SECURITY_AND_RESOURCE_GOVERNANCE.md`](docs/SECURITY_AND_RESOURCE_GOVERNANCE.md)
- [`docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md`](docs/CRATE_ARCHITECTURE_AND_DEPENDENCIES.md)

### Execution program

- [`docs/WORKSTREAM_GRAPH.md`](docs/WORKSTREAM_GRAPH.md)
- [`docs/FIRST_IMPLEMENTATION_CAMPAIGN.md`](docs/FIRST_IMPLEMENTATION_CAMPAIGN.md)
- [`docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md`](docs/RISK_REGISTER_AND_RESEARCH_AGENDA.md)

### Machine-readable registries

- [`registries/compatibility_profiles.toml`](registries/compatibility_profiles.toml)
- [`registries/evidence_classes.toml`](registries/evidence_classes.toml)
- [`registries/workstreams.toml`](registries/workstreams.toml)
- [`registries/claims.toml`](registries/claims.toml)

---

## 49. Immediate next actions

The first implementation sequence is:

1. build the registry/claim/work-graph validators;
2. freeze the exact initial CPython/platform profile matrix;
3. build the isolated SymPy 1.14.0 conformance runner and initial object inventory;
4. implement IDs, outcomes, budgets, `Cx`, canonical encoding, and exact arithmetic reference lanes;
5. implement terms/domains/contexts/bindings and proof-kernel nucleus;
6. implement the initial Python shell and custom-subclass/held-form corpus;
7. implement polynomial representations, identity verifier, and factorization portfolio;
8. integrate structured cancellation, replay, and typed continuation;
9. implement proof-producing differentiation and certified/reference numeric compilation;
10. add persistence/checkpoint/RaptorQ and agent-protocol slices;
11. run and close the Certified Jacobian Pipeline;
12. only then expand broad SymPy surface work.

The first review should be adversarial: find a real SymPy object behavior the dual-lane shell cannot preserve, a claim whose evidence class is too strong, a verifier that is not independent, or a subsystem whose complexity is not earned.

---

## 50. Final design principle

The project should optimize for a future in which symbolic systems are not opaque oracles but inspectable mathematical execution environments.

A successful FrankenSymPy result should be able to answer:

- What exactly is the value?
- Under which domain, assumptions, and branch policy?
- Is it profile-compatible, mathematically proved, certificate-verified, numerically certified, heuristic, conditional, or inconclusive?
- Which algorithms were considered and why was this one chosen?
- What did it cost and what budget remains?
- Can it be cancelled, resumed, replayed, verified elsewhere, or merged into another derivation?
- Can an agent operate on it structurally without parsing prose?
- Can a corrupted checkpoint be repaired without confusing recovered bytes for truth?
- Can every public claim be recomputed from evidence on the release commit?

That is the radical leapfrog: not merely faster symbolic manipulation, but a symbolic mathematics substrate whose compatibility, truth, execution, collaboration, and claims are all first-class and machine-checkable.