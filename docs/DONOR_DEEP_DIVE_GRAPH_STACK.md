# Donor deep dive: FrankenGraphDB and FrankenNetworkX

**Status:** normative source audit and architecture input  
**Pinned sources:** `Dicklesworthstone/frankengraphdb@df733010a46ac2a725df141204d7940b4e37d4b7`, `Dicklesworthstone/franken_networkx@972814b1b4649c20b6f2acdd7456e3580cefdbeb`  
**Audit date:** 2026-08-20  
**Scope:** deterministic graph semantics, proof and dependency graphs, content-addressed history, branches, claim/evidence lattices, purpose-typed authority, algorithm witnesses, conformance, and graph-index projections

## 1. Executive conclusion

The shallow interpretation of the graph donors is “use graph algorithms somewhere in the planner.” The useful inheritance is much larger and more disciplined.

FrankenSymPy is itself a system of interlocking graphs:

- immutable term DAGs;
- binder and substitution dependency graphs;
- proof and certificate DAGs;
- assumptions, implication, and contradiction graphs;
- rewrite-rule applicability and critical-pair graphs;
- domain/coercion graphs;
- algorithm-plan and fallback graphs;
- workspace branch and merge graphs;
- artifact, provenance, and invalidation graphs;
- implementation workstream and release-gate DAGs.

FrankenNetworkX contributes a deterministic local graph-algorithm discipline: explicit tie-break policies, insertion-order fidelity where required, decision-path witnesses, adversarial tie-break corpora, and honest surface accounting. FrankenGraphDB contributes the larger artifact/history architecture: immutable content-addressed objects, append-only commit lineage, branches over shared state, claim/evidence class separation, purpose-typed capability contexts, and rebuildable derived graph indexes.

The architectural conclusion is:

> FrankenSymPy owns a small, safe, deterministic graph substrate in its semantic core. FrankenGraphDB may index and query very large mathematical histories, but no graph database row, index, traversal, score, or plan certificate is mathematical authority.

## 2. Source surfaces examined

### FrankenNetworkX

- `README.md`: compatibility doctrine, feature inventory, conformance gates, and performance evidence discipline;
- `crates/fnx-cgse/src/lib.rs`: closed tie-break policies, complexity witnesses, decision-path hashing, and witness ledger;
- `crates/fnx-conformance/`: differential and adversarial conformance, counterexample mining, and tie-break corpora;
- generated coverage/delegation/divergence ledgers and backend-dispatch inventory;
- graph-type storage and insertion-order behavior documented in the public contract.

### FrankenGraphDB

- `README.md`: Chronicle, branches, deterministic plan certificates, graph-structured storage, and rebuildable derived structures;
- `Cargo.toml`: closed dependency universe and workspace safety policy;
- `crates/fgdb-types/src/context.rs`: purpose-typed contexts and affine obligations;
- `crates/fgdb-claim/src/lib.rs`: claim/evidence strength lattice with compile-time and runtime enforcement;
- `crates/fgdb-chronicle/src/{capsule,commit,identity,marker,pack,root,scrub,store,symbol}.rs`: immutable objects, commit publication, marker discipline, packing, scrubbing, and recovery;
- pure canonical type and identity surfaces in `fgdb-types`.

The audit treats current implementations as evidence about mechanisms, not as blanket dependencies or proof of suitability for every symbolic workload.

## 3. Adopt from FrankenNetworkX

### 3.1 Explicit canonical tie-breaks

Many symbolic operations have multiple mathematically valid outputs:

- factor ordering;
- basis ordering;
- dummy-symbol naming;
- term traversal order;
- equal-cost rewrite choice;
- proof-search frontier order;
- canonical representative selection inside an orbit;
- ordering of independent derivation alternatives.

Leaving these choices to hash iteration, thread scheduling, allocator addresses, or incidental traversal order destroys reproducibility and compatibility.

FrankenSymPy therefore defines a closed registry of tie-break policies. A public operation either:

1. inherits the exact policy of the active compatibility profile;
2. uses a named native policy whose observable difference is declared; or
3. refuses strict reproducibility because no policy has been certified.

The policy identifier is part of the decision receipt and replay closure. It is not necessarily part of mathematical term identity; that depends on whether the selected ordering is semantically canonical for the object kind.

### 3.2 Decision-path witnesses

A graph algorithm may emit a compact witness containing:

- graph snapshot root;
- algorithm and policy identifiers;
- input roots and dimensions;
- seed lineage where applicable;
- dominant complexity term and bounded counters;
- ordered decision-path digest;
- selected output root;
- verifier profile for any checkable graph certificate.

This is operational evidence. A digest that two runs made the same choices does not prove a theorem. Where the graph result matters to correctness, a small checker validates the corresponding certificate.

### 3.3 Honest compatibility inventory

FrankenNetworkX’s present/partial/missing discipline maps directly to the SymPy surface. FrankenSymPy must never turn “importable name” into “behaviorally compatible implementation.” Every Python-facing surface is classified by exact profile and evidence level.

### 3.4 Adversarial tie-break corpus

Each canonicalization-sensitive family receives fixtures that maximize ambiguity:

- automorphic graphs and symmetric expressions;
- equal weights and equal degrees;
- duplicate structural hashes with different provenance;
- equivalent factorizations or bases in different insertion orders;
- randomized thread schedules;
- stable-label collisions at coarse summary levels;
- graph mutations that should and should not invalidate a result.

A green happy-path corpus cannot establish determinism.

## 4. Adapt from FrankenGraphDB

### 4.1 One immutable object universe

Terms, domains, contexts, claims, certificates, proof nodes, counterexamples, decisions, and benchmark fixtures are immutable content-addressed objects. Mutable authority is a small set of roots:

- workspace branch heads;
- active compatibility/profile roots;
- accepted claim manifests;
- registry generations;
- release manifests.

The object store can be memory-only, file-backed, browser-backed, FrankenSQLite-backed, or remotely replicated. Identity and verification semantics are storage-neutral.

### 4.2 Branch-per-agent without branch-per-truth

Agents may fork O(1)-style logical branches over shared immutable content and publish candidate derivations independently. A branch permits isolation and comparison; it does not create a second mathematical truth.

Merging branches follows the symbolic transaction contract:

1. identical content deduplicates;
2. disjoint authoritative writes merge;
3. registered commutative intents may merge with a certificate;
4. pure intents may be deterministically replayed against a new base;
5. all other cases remain explicit conflicts or separate branches.

### 4.3 Claim/evidence lattice

The graph stack’s strongest reusable governance idea is that weaker evidence cannot justify a stronger claim. FrankenSymPy maintains distinct classes such as:

- invariant;
- theorem/proof within a named model;
- bounded model check;
- exact certificate verification;
- compatibility observation;
- statistical calibration claim;
- service-level objective;
- benchmark measurement;
- research hypothesis.

The lattice is enforced in registries and, for critical Rust APIs, by sealed marker types. A benchmark cannot justify a correctness invariant. A decision witness cannot justify a factorization claim. A signature cannot justify mathematical truth.

### 4.4 Purpose-typed authority and obligations

Graph queries, workspace transactions, verification, publication, maintenance, and remote replication receive distinct capability wrappers. The portable mathematical verifier remains runtime-independent, but hosted orchestration must not pass a root context with ambient authority into every subsystem.

Long-lived obligations are affine and auditable:

- snapshot pins;
- workspace leases;
- prepared publication bytes;
- remote subgoal leases;
- temporary artifact roots;
- verifier permits;
- branch publication reservations.

Cancellation must discharge or explicitly transfer every obligation before a region closes.

### 4.5 Rebuildable projections

FrankenGraphDB may provide high-scale projections for:

- proof search and dependency exploration;
- provenance and “why did this rebuild?” queries;
- impact cones;
- counterexample similarity;
- branch comparison;
- rule-use analytics;
- corpus-scale graph mining;
- hybrid text/vector/graph retrieval.

The projection is derived. Deleting and rebuilding it from authoritative object roots must preserve every accepted result. An index miss is never evidence that a proof, rule, or dependency does not exist unless the complete authoritative closure and index-completeness certificate are present.

## 5. Native graph families in FrankenSymPy

### 5.1 Term DAG

Nodes are canonical term objects; edges identify ordered child positions and binder structure. Required operations include:

- topological traversal;
- structural hashing and equality support;
- subterm reachability;
- common-subexpression closure;
- dependency slices;
- cycle refusal in formats that require DAGs;
- bounded graph serialization.

### 5.2 Proof and certificate DAG

Nodes include claims, lemmas, certificate steps, imported axioms, and verifier dependencies. Required operations include:

- complete dependency closure;
- proof minimization candidates;
- dominator and articulation analysis for fragility;
- unresolved leaf detection;
- independent-checker partitioning;
- provenance queries;
- certificate chunk prioritization.

A proof DAG operation proposes a slice. The domain verifier checks the slice’s logical sufficiency.

### 5.3 Assumption and invalidation graph

Edges distinguish implication, dependency, contradiction, refinement, and inherited context membership. Required operations include:

- incremental invalidation cones;
- contradiction witness paths;
- context-fork delta computation;
- strongly connected equivalence regions where justified;
- conservative “may depend” summaries with no false negatives;
- exact refinement before publication.

### 5.4 Rule and critical-pair graph

Nodes are rewrite rules and overlap regions. Edges indicate potential critical pairs, ordering constraints, generated lemmas, and subsumption. Uses include:

- confluence research;
- rule scheduling;
- loop-risk detection;
- e-graph extraction support;
- proof obligation generation.

Graph analysis never upgrades an unproved rewrite into a trusted identity.

### 5.5 Domain and coercion graph

Nodes are exact domains and extension worlds; directed edges are certified coercions or embeddings. The graph supports:

- path selection under exact policy;
- ambiguity detection;
- coherence checks;
- least-common-domain candidates;
- cycle and lossiness diagnostics.

A selected path is checked against registered coercion laws and profile semantics.

### 5.6 Algorithm and work graph

The planner’s strategy portfolio and the repository’s implementation plan are both DAGs. The graph substrate provides:

- dependency validation;
- acyclicity checks;
- critical path and frontier computation;
- budgeted scheduling candidates;
- cancellation/drain ownership;
- release-gate closure.

This graph is operational and cannot alter mathematical canonical results.

## 6. Dependency-light graph core

The first graph crate should be small and safe:

```text
fsym-graph-types
├── compact stable node/edge identifiers
├── deterministic adjacency representation
├── bounded builders and decoders
└── no Python, runtime, database, network, or planner

fsym-graph-core
├── topological order and cycle witness
├── SCC and condensation
├── reachability and path witness
├── dominators / dependency cuts
├── bounded transitive closure helpers
├── deterministic traversal policies
└── certificate emitters and reference checkers
```

Higher layers may adapt mature `fnx-*` crates where their dependency and compatibility contracts satisfy FrankenSymPy’s closed-universe policy. The verifier-facing subset must remain independently embeddable and must not require a Python graph object.

## 7. Graph certificate families

Initial checkable certificate families:

| Certificate | Claim checked by a small verifier |
|---|---|
| topological order | every edge advances in the supplied order and every node occurs once |
| cycle witness | supplied directed edges form a closed valid cycle |
| SCC partition | membership is total/disjoint and mutual reachability certificates support each component |
| condensation | quotient edges match source graph component crossings |
| reachability path | every adjacent pair is an edge and endpoints match |
| dominator witness | dominator tree and semidominator evidence satisfy the named algorithm contract |
| invalidation slice | every marked object is reachable from a changed authoritative root under declared edge kinds |
| cut witness | removing the supplied cut separates the named source and target sets |
| canonical traversal | output follows the registered tie-break policy over the exact snapshot |

Not every optimized algorithm needs to ship a full proof. The release matrix decides which result classes require certificates, differential checking, or replay witnesses.

## 8. Performance inheritance

The graph donors reinforce several performance rules:

- compact integer handles inside one snapshot; durable IDs only at boundaries;
- adjacency and attributes stored together when access patterns demand it;
- avoid per-edge hash lookups in hot kernels;
- preserve insertion order explicitly rather than reconstructing it;
- separate scalar reference and optimized routes;
- benchmark complete operations, including Python conversion and attribute access;
- measure multiple graph regimes, not one friendly topology;
- instrument cache misses, allocation, branch behavior, and tail latency;
- require same-invocation incumbent comparisons and negative controls.

FrankenSymPy applies these to term/proof graphs and to ordinary symbolic kernels whose bottleneck is repeated indirect lookup.

## 9. Research lanes

Promising but non-foundational lanes include:

- dynamic graph algorithms for enormous evolving proof corpora;
- spectral or topological health signals for rule systems;
- graph neural retrieval over derivation histories;
- minimum proof-cut and explanation optimization;
- proof-carrying distributed graph analytics;
- RaptorQ-native graph artifact exchange;
- graph-aware Slepian–Wolf deltas between nearby branches;
- sheaf and local-to-global diagnostics with precisely stated models.

Each remains optional until a corpus, baseline, failure model, and acceptance criterion are registered.

## 10. Explicit rejections

FrankenSymPy rejects:

- a graph database as the source of mathematical identity;
- hash-map iteration as a canonical order;
- centrality, similarity, or model score as proof evidence;
- “no path found” from an incomplete derived index as a negative theorem;
- raw byte merge of serialized terms or proofs;
- graph certificates that merely restate an algorithm’s output without an independent check;
- importing the full Python NetworkX surface into portable verifier crates;
- hidden nondeterminism justified as an implementation detail;
- graph analytics that can mutate accepted state without a workspace transaction.

## 11. Implementation order

1. freeze graph object and tie-break registries;
2. implement bounded deterministic graph types;
3. add topological, cycle, SCC, reachability, and invalidation reference algorithms;
4. define portable graph certificate schemas and mutation corpora;
5. add optimized safe-Rust kernels with differential checks;
6. integrate proof, assumptions, rule, and workstream graph views;
7. add optional FrankenGraphDB projection adapters;
8. add branch/provenance graph queries;
9. certify performance by regime;
10. admit research lanes only after their gates exist.
