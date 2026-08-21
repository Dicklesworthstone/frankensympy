# Deterministic graph reasoning substrate

**Status:** normative architecture contract  
**Scope:** graph identities, snapshot semantics, canonical traversal, local algorithms, portable certificates, derived projections, resource limits, and trust boundaries

## 1. Contract

FrankenSymPy SHALL provide a dependency-light graph substrate for semantic graphs that are intrinsic to symbolic computation. The substrate is not a general database and is not the mathematical verifier itself.

The required separation is:

```text
immutable semantic objects
        │
        ▼
snapshot-bound graph views
        │
        ├── deterministic local graph algorithms
        ├── certificate generation
        ├── optimized indexes and projections
        └── optional remote analytics
                  │
                  ▼
      small independent certificate verifier
                  │
                  ▼
       domain-specific mathematical verifier
```

A graph result can establish only the graph claim represented by its certificate. It cannot establish algebraic equality, analytic validity, satisfiability, or theoremhood unless a domain verifier separately checks the mathematical reduction.

## 2. Identity layers

### 2.1 Durable object identity

Every authoritative node is an immutable content-addressed object with an object-kind domain separator. Durable identity does not contain:

- arena address;
- process or thread ID;
- graph insertion slot;
- database row ID;
- transport chunk position;
- runtime task ID;
- wall-clock timestamp.

### 2.2 Snapshot-local handles

Within one immutable graph snapshot, nodes and edges use compact integer handles. Handle assignment is deterministic for a canonical snapshot encoding but is not a cross-snapshot object identity.

### 2.3 Graph snapshot root

A graph snapshot root commits to:

- graph schema and version;
- node object IDs and kinds;
- typed ordered edges;
- edge attributes that affect semantics;
- canonical ordering policy;
- universe/profile roots required to interpret nodes and edges;
- optional completeness declaration.

Derived indexes, compressed layouts, and physical chunking are excluded from semantic identity.

## 3. Edge kinds

Edge meaning is explicit. Initial families include:

- `ordered_child(position)`;
- `depends_on`;
- `proves`;
- `uses_rule`;
- `uses_assumption`;
- `implies`;
- `contradicts`;
- `refines`;
- `coerces_to`;
- `invalidates`;
- `generated_from`;
- `alternative_derivation`;
- `must_precede`;
- `may_overlap`;
- `branch_parent`;
- `artifact_contains`.

Algorithms declare the accepted edge-kind set. Treating all edges as interchangeable is a schema error.

## 4. Canonical ordering

Every algorithm that can return multiple equivalent outputs declares a tie-break policy from `registries/graph_reasoning.toml`.

Strict compatibility may require insertion order inherited from a reference profile. Native deterministic modes should prefer content-based total orders that remain stable across serialization and process boundaries.

A policy includes:

- stable identifier and version;
- applicable node/edge kinds;
- primary and secondary comparison keys;
- treatment of incomparable or missing keys;
- whether insertion order participates;
- seed derivation if randomized but reproducible;
- compatibility profiles in which it is valid.

A policy change is a compatibility change and invalidates decision receipts that name the old policy.

## 5. Required local algorithms

### 5.1 Graph construction and validation

- bounded canonical decode;
- duplicate-node and duplicate-edge policy enforcement;
- endpoint and kind validation;
- DAG-required cycle refusal;
- complete input consumption;
- unknown critical-field refusal;
- deterministic normalization.

### 5.2 Reachability

- single-source and multi-source reachability;
- path reconstruction;
- reverse dependency cone;
- bounded early-stop queries;
- exact negative result only when snapshot completeness is established.

### 5.3 DAG algorithms

- topological ordering;
- cycle witness on failure;
- longest/critical dependency path for operational graphs;
- incremental invalidation frontier;
- antichain and ready-frontier extraction;
- transitive reduction candidates where semantically safe.

### 5.4 General directed graph algorithms

- strongly connected components;
- condensation DAG;
- dominators and post-dominators;
- articulation/cut diagnostics on appropriate projections;
- bounded shortest paths under exact nonnegative integer weights;
- deterministic traversal trees.

### 5.5 Proof/provenance helpers

- dependency closure;
- minimum explanation candidate under a registered cost model;
- alternate-derivation enumeration with budget;
- unresolved leaf set;
- provenance slice;
- fragility/dominator report.

Cost-model optimization is advisory unless a checker verifies the exact objective and constraints.

## 6. Resource totality

All artifact-facing graph APIs accept explicit limits:

- maximum input bytes;
- nodes and edges;
- nesting and attribute bytes;
- work/fuel units;
- queue/frontier size;
- certificate bytes;
- output rows;
- recursion depth where recursion remains;
- wall-time only in hosted wrappers, never as a portable semantic result.

Exhaustion returns `Inconclusive(ResourceLimit { dimension, observed, limit })`. It is not an empty graph, a negative reachability result, or a malformed-input verdict.

## 7. Portable verifier API

Reference APIs are synchronous and runtime-free:

```rust
pub fn verify_topological_order(
    graph: &GraphView<'_>,
    certificate: &TopologicalOrderCertificate,
    limits: VerificationLimits,
) -> Result<VerifiedTopologicalOrder, GraphVerificationError>;

pub fn verify_reachability_path(
    graph: &GraphView<'_>,
    certificate: &ReachabilityPathCertificate,
    limits: VerificationLimits,
) -> Result<VerifiedPath, GraphVerificationError>;
```

The verifier crates SHALL NOT require:

- the algorithm that generated the certificate;
- asupersync;
- Python;
- FrankenGraphDB;
- a persistent store;
- networking;
- planner or telemetry crates;
- graph visualization.

## 8. Snapshot completeness

Negative graph claims require a completeness basis. A certificate that “no dependency path exists” must bind to a graph snapshot whose manifest asserts and verifies the complete relevant edge closure.

Completeness may be scoped:

- entire proof closure;
- one registry generation;
- one assumptions context;
- one namespace prefix;
- one workstream DAG;
- one branch interval.

An index or remote query result without this basis can return only `NotObserved`, never `DoesNotExist`.

## 9. Derived projection adapter

The optional FrankenGraphDB adapter may:

- ingest immutable objects and typed edges;
- maintain text/vector/graph search indexes;
- answer large-scale provenance and impact queries;
- store branch and event projections;
- cache graph algorithm outputs;
- produce candidate certificates or witness paths.

It may not:

- mint semantic object IDs;
- admit claims;
- mutate assumptions or registries outside a workspace transaction;
- upgrade evidence;
- make an incomplete negative result authoritative;
- replace the portable graph or mathematical verifier.

Every authoritative projection read binds to an ingestion watermark and source manifest root.

## 10. Invalidation

Changes publish typed invalidation roots. The graph substrate computes a conservative reverse-reachability slice. Exact cache invalidation requires:

- complete dependency edges for the cached result;
- exact universe and profile roots;
- no pruned validation interval;
- matching graph schema and edge semantics.

Approximate summaries may enlarge invalidation. They may never omit a true dependent object.

## 11. Conformance

Required suites include:

- canonical encode/decode round trips;
- arbitrary chunk-boundary streaming;
- malformed and over-budget decodes;
- permutation tests for policy-allowed invariance;
- insertion-order tests for profiles where order is observable;
- thread-count and schedule replay;
- adversarial symmetric graphs;
- duplicate and collision fixtures;
- differential tests against scalar reference algorithms;
- certificate mutation tests;
- incomplete-index negative-result tests;
- projection rebuild equivalence.

## 12. Performance gates

Performance claims are separated by graph regime:

- shallow wide DAG;
- deep narrow DAG;
- high sharing term DAG;
- cyclic rule graph;
- scale-free provenance graph;
- dense small graph;
- enormous sparse graph;
- attribute-heavy compatibility graph;
- incremental mutation stream.

Gates include complete operation cost, peak memory, allocation count, tails, cancellation/drain cost, and certificate overhead. No graph speed claim is inferred from a kernel that omits attribute lookup, Python conversion, or verification when the public operation includes those costs.

## 13. Safety and dependencies

Project-authored graph core and verifier crates use `#![forbid(unsafe_code)]`. External algorithmic dependencies are rejected unless admitted through the project dependency registry with a narrower and better-audited contract than implementing the required algorithm in the FrankenSuite.

FrankenNetworkX crates may be reused selectively after dependency, safety, determinism, and API review. “Same owner” is not automatic admission.

## 14. Release blockers

A release profile that claims deterministic or certified graph behavior is blocked by:

- unregistered tie-break choice;
- schedule-dependent output;
- missing snapshot root;
- negative claim from incomplete closure;
- verifier depending on generator/runtime/database;
- unknown critical graph schema fields;
- unbounded decode or allocation from hostile counts;
- graph projection treated as authoritative identity;
- certificate accepted without mutation-negative tests;
- documentation wording stronger than registry evidence.
