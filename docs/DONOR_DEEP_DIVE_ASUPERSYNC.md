# Donor deep dive: asupersync and the Asupersync Transfer Protocol

**Status:** normative source audit and architecture input  
**Pinned source:** `Dicklesworthstone/asupersync@3cb2dc6d0540a961345e61bc9f89b4e930cbb8cf`  
**Audit date:** 2026-08-20  
**Scope:** structured concurrency, capabilities, deterministic lab execution, ATP object graphs, manifests, journals, proof bundles, delta transfer, repair, governance, and statistical monitoring

## 1. Executive conclusion

The first FrankenSymPy plan inherited asupersync mostly as “the async runtime plus RaptorQ.” That was far too shallow.

Asupersync contains a broader family of mechanisms that map unusually well onto symbolic computation:

- capability-scoped effect authority;
- owned region trees and loser draining;
- deterministic lab/live differential execution;
- verified object graphs rather than flat files;
- canonical manifests and Merkle closure;
- staged append-only journals with resumable durable evidence;
- offline proof bundles;
- fail-closed verifier pipelines;
- preflight plans and execution-deviation reports;
- content-defined delta synchronization and CAS reuse;
- side-information-aware Slepian–Wolf delta coding;
- multi-source usefulness-aware scheduling;
- explicit resource governance and fairness;
- adaptive repair economics;
- conformal calibration and anytime-valid e-process monitoring.

The most accretive design change is therefore not “use ATP to send checkpoints.” It is to give FrankenSymPy a **portable mathematical artifact plane** whose semantics are inspired by ATP but whose trusted core remains independent of networking and the runtime.

## 2. Source surfaces examined

The audit examined at least these pinned surfaces:

| Surface | Relevant source |
|---|---|
| capability context | `src/cx/cx.rs` |
| structured regions and spawning | `src/cx/scope.rs` |
| ATP module topology | `src/atp/mod.rs` |
| verified object identity | `src/atp/object.rs` |
| canonical manifests | `src/atp/manifest.rs` |
| preflight planner | `src/atp/planner.rs` |
| transfer priority and utility | `src/atp/transfer_brain.rs` |
| deterministic resource governance | `src/atp/governance/mod.rs` |
| append-only resume journal | `src/atp/journal/append_journal.rs` |
| delta CAS and Merkle diffs | `src/atp/journal/delta_cas.rs` |
| persistent content-defined delta path | `src/atp/delta.rs` |
| Slepian–Wolf side-information delta | `src/atp/slepian_wolf.rs` |
| offline proof bundle | `src/atp/proof/bundle.rs` |
| fail-closed verifier pipeline | `src/atp/verifier.rs` |
| multi-source repair scheduler | `src/atp/repair_scheduler.rs` |
| adaptive repair policy | `src/atp/adaptive_raptorq.rs` |
| lab/live replay identity | `src/lab/dual_run.rs` |
| conformal calibration | `src/lab/conformal.rs` |
| anytime regression monitoring | `src/raptorq/regression.rs` |
| obligation e-process | `src/obligation/eprocess.rs` |
| latency hedging | `src/combinator/hedge.rs` |

This audit distinguishes reusable implementation from architectural precedent. No source file is treated as automatically production-ready for FrankenSymPy.

## 3. Capability contexts become mathematical authority types

Asupersync's `Cx` is more than a cancellation token. It combines type-level and runtime capability narrowing for spawn, time, entropy, I/O, and remote authority, while carrying cancellation, budgets, trace identity, and optional evidence sinks.

FrankenSymPy should expose purpose-built restricted contexts:

```text
PureVerifierCx
├── exact arithmetic fuel
├── allocation/proof/output budgets
├── deterministic trace sink, optional
└── no spawn/time/entropy/io/remote/python/persistence/publication

LocalGeneratorCx
├── owned spawn
├── deterministic entropy streams
├── local cache read
└── no remote or publication unless separately granted

RemoteGeneratorCx
├── immutable packet read
├── leased seed/subgoal range
└── candidate-response write only

PythonHookCx
├── supervised interpreter call
├── nested callback/output budget
└── no theorem-grant authority

PublisherCx
├── verified result object
├── atomic cache/ledger publication
└── no generator authority
```

The key inheritance is **authority separation**. A component should be unable—not merely instructed not—to perform effects outside its role.

The portable verifier remains a pure function and does not depend on asupersync. Service wrappers may translate a restricted `Cx` budget into verifier limits.

## 4. Region ownership becomes proof-work ownership

Asupersync regions own child tasks, finalizers, pending spawns, cancellation, and close. This maps directly onto symbolic portfolios:

```text
mathematical request region
├── diagnostic probes
├── candidate generator regions
│   ├── modular factorization
│   ├── sparse interpolation
│   └── deterministic baseline
├── verifier region
├── checkpoint/artifact region
└── publication finalizer
```

Adopt:

- no detached simplification, proof search, cache warming, or remote leases;
- pending-spawn accounting before region close;
- loser cancellation followed by draining;
- finalizers that can prove permits, leases, temporary objects, and publication intents are balanced;
- child budgets that cannot consume verifier reserves;
- region-owned immutable term handles rather than borrowed Python state.

Adapt:

- a task completing first may nominate a candidate but cannot choose a strict canonical result;
- verifier success is a separate publication phase;
- exact term/proof identity never includes runtime region or task IDs;
- non-cooperative Python hooks are a typed boundary, not covered by universal drain-latency claims.

## 5. ATP verified object graphs become mathematical artifact graphs

ATP explicitly moves verified object graphs, not merely files. It separates content-addressed IDs, manifest-addressed IDs, canonical graph structure, and metadata policy.

FrankenSymPy should use a mathematical object graph with kinds such as:

- term;
- operator definition;
- domain;
- assumptions context;
- branch policy;
- claim;
- certificate;
- proof node;
- registry fragment;
- verifier manifest;
- counterexample;
- decision and execution receipt;
- checkpoint state;
- generated program;
- compatibility observation;
- benchmark or mutation fixture.

Adopt:

- domain-separated content IDs;
- streaming content hashing;
- separate immutable content identity from mutable manifest/workspace identity;
- canonical object ordering;
- explicit object kinds;
- manifest roots over the complete reachable closure.

Reject:

- using ATP `ContentId` directly as `TermId`;
- treating a transport manifest as a mathematical proof;
- allowing metadata policy or chunk layout to affect semantic identity.

## 6. Canonical manifests become verifier-complete capsules

ATP manifests are versioned, deterministic, self-describing, Merkle-bound, and distinguish critical from optional fields.

FrankenSymPy should apply the same discipline to `Claim + Certificate` capsules:

- critical unknown claim, domain, context, proof, identity, or resource fields fail closed;
- optional provenance may be preserved without affecting truth;
- canonical encoding is language-neutral;
- Merkle closure names every authoritative dependency;
- chunking, compression, encryption, repair, and placement are transport metadata outside mathematical identity;
- canonical NaN, signed-zero, integer, map, and sequence encodings are specified rather than delegated to incidental serializers.

The manifest enables an offline verifier to answer: “Do I possess the complete exact universe required to check this claim?”

## 7. ATP journals become mathematical publication journals

The append journal distinguishes:

```text
Offer
Accept
ChunkReceived
ChunkVerified
ChunkWritten
RepairDecode
CommitIntent
CommitComplete
Cancellation
Rollback
CompactionBoundary
ProofDigest
```

A symbolic publication journal should distinguish:

```text
RequestAccepted
CandidateProduced
CandidateCanonicalized
CertificateProduced
CertificateVerified
ArtifactDurable
PublicationIntent
PublicationComplete
Cancellation
Rollback
CompactionBoundary
VerificationDigest
```

Important inherited rule:

> Progress is resumable only when the evidence required for that stage is both verified and durable.

A candidate in memory is not a durable subproof. Bytes written without verification are not reusable mathematical progress. A verification record without the exact object bytes and universe closure is not replayable.

The journal should compute resumable summaries over verified subclaims and exact frontier state, analogous to ATP's durable chunk summary.

## 8. Offline ATP proof bundles become portable mathematical proof bundles

ATP proof bundles package manifest roots, object roots, verification evidence, repair metadata, journal digests, replay pointers, and extensible artifacts for offline auditing.

FrankenSymPy should define bundle strengths by **closure**, not by one scalar “proof strength”:

- `VerifierComplete`: enough to check the typed claim;
- `ReplayComplete`: verifier closure plus generator decision, seed, and continuation state;
- `CompatibilityComplete`: exact profile/environment observations;
- `PublicationComplete`: verifier closure plus durable publication and signatures;
- `ResearchComplete`: search trace and rejected candidates, explicitly non-authoritative.

Bundles should be independently verifiable and transferable. Signatures establish origin. RaptorQ establishes byte recoverability within a loss model. Neither establishes mathematical truth.

One ATP detail should not be copied blindly: a serialization round trip that drops a commit object or other authoritative field is unacceptable for a verifier-complete FrankenSymPy bundle. Every omitted field must be explicitly optional and reconstructible from content-addressed dependencies.

## 9. ATP's verifier pipeline strengthens the acceptance boundary

ATP's verifier is deliberately independent from its writer and journal. It bounds inputs, checks staged identities, verifies proof bundles and finalizer evidence, and records whether output was exposed before commit.

This directly supports the public FrankenSymPy promise:

- verifier crates are independently embeddable;
- verifier decoders are bounded before allocation;
- verifier stages have stable names and typed rejection reasons;
- finalizer evidence proves workers and permits are drained/balanced;
- output exposure before verification is a gate failure;
- cancellation may preserve a verifier continuation without granting a result.

The crucial adaptation is to keep the mathematical verifier **even smaller** than ATP's hosted pipeline: no async runtime, writer, journal, network, or persistent store in the portable crate.

## 10. ATP preflight planning becomes `plan_only`

ATP plans before irreversible network, relay, disk, or sync effects and records uncertainty and later deviations.

FrankenSymPy should expose:

```text
plan_only(operation)
├── inferred domains and unresolved ambiguities
├── problem dimensions and structural diagnostics
├── eligible and rejected algorithms
├── expected exact arithmetic/proof/output growth
├── verifier cost and protected budget
├── reusable cached subclaims
├── checkpoint opportunities
├── remote/persistence requirements
├── expected evidence class
├── uncertainties and fallback graph
└── no execution side effects
```

Execution produces a deviation report:

- expected versus actual representation;
- predicted versus actual coefficient/proof growth;
- strategy launches and cancellations;
- verifier rejection;
- checkpoint and cache decisions;
- resource-estimate error;
- reason for fallback.

A plan receipt explains execution. It is not mathematical evidence.

## 11. Content-defined delta transfer becomes semantic delta synchronization

ATP combines content-defined chunks, content-addressed stores, Merkle manifests, receiver have-sets, sub-chunk operations, and changed-subtree discovery.

FrankenSymPy can use three delta levels:

1. **Semantic object delta:** transfer only new `TermId`, `ProofId`, `DomainId`, and registry objects.
2. **Container delta:** chunk huge coefficient arrays, sparse matrices, proof tables, and generated programs along semantic boundaries.
3. **Byte delta:** use content-defined chunks for opaque large blobs.

This enables:

- agent branches to exchange only changed derivation subgraphs;
- remote workers to avoid resending known terms and domains;
- proof archives to deduplicate shared lemmas;
- checkpoint updates to transfer changed frontier blocks;
- ecosystem conformance runs to reuse unchanged fixture objects.

The logical object manifest remains independent of physical chunking. A new chunker must not change term or proof identity.

## 12. Slepian–Wolf coding is a research-grade optimization, not a foundation

ATP's Slepian–Wolf path uses receiver side information to encode only uncertainty in a localized changed byte region.

Potential FrankenSymPy use:

- closely related large modular matrices;
- successive Gröbner checkpoints;
- evolving proof tables;
- near-identical generated code or conformance artifacts;
- branch states where both peers share an older verified object.

This can approach the information-theoretic delta floor when side information is excellent.

Constraints:

- optional artifact-transfer layer only;
- never part of claim, term, or certificate semantics;
- deterministic frame derivation and bounded decode;
- exact post-decode content hash required;
- benchmarked against ordinary semantic-object and content-defined delta;
- disabled unless expected transfer savings exceed CPU, complexity, and failure costs.

It is a promising alien-artifact feature, but it must earn itself with a real corpus.

## 13. Transfer Brain priorities become proof-availability priorities

ATP optimizes verified completion and early usability, not raw throughput. Its priorities distinguish control, early usability, decode usefulness, standard data, repair, and speculation.

A symbolic artifact scheduler should prioritize:

1. manifest, claim schema, verifier profile, and domain/context roots;
2. certificate nodes that unlock the earliest complete subclaim;
3. source terms needed by those verifier steps;
4. remaining verifier-complete closure;
5. replay/search/provenance extras;
6. repair symbols;
7. speculative neighboring artifacts.

This creates a **proof availability frontier**: the receiver learns not merely “60% of bytes arrived” but “these exact claims are already independently checkable.”

Metrics should include:

- time to first verified subclaim;
- time to verifier-complete result;
- bytes transferred before verification;
- proof-unlocking value per byte;
- duplicate and cancelled bytes;
- verifier CPU per accepted claim;
- resume value;
- repair ROI.

## 14. Resource governance becomes mathematical workload governance

ATP's governor is explicit, deterministic, and side-effect free; it evaluates demands against a budget and supports fairness across concurrent transfers.

FrankenSymPy should separate:

- user-visible request priority;
- verifier-protected capacity;
- generator and speculative capacity;
- proof/archive transfer capacity;
- persistence and repair capacity;
- tenant/agent fair shares.

A service must reserve verifier memory and CPU before launching generators. Interactive small proofs should not starve behind one enormous Gröbner basis. Best-effort cache seeding and proof compression yield first.

Fairness policy is operational and cannot alter the mathematical canonical result.

## 15. Multi-source repair scheduling becomes untrusted subclaim scheduling

ATP scores peers by path quality, budget, symbol rarity, decode usefulness, trust, relay cost, and churn; it rejects stale, duplicate, wrong-group, wrong-transfer, unauthenticated, low-value, or malicious symbols.

For distributed symbolic work:

- schedule rare or proof-unlocking subclaims first;
- bind every response to packet, claim, domain, context, registry, and seed lease;
- cap per-worker batches to avoid monopoly;
- retain bounded rejection history plus lifetime counts;
- distinguish unavailable, malformed, wrong-universe, duplicate, low-usefulness, and verifier-rejected work;
- never let an unauthorized response cancel an honest worker's lease;
- use worker history only for scheduling, never for mathematical trust;
- verify locally or with an independently trusted verifier.

A worker can make the system faster. It cannot make an invalid result true.

## 16. Adaptive repair becomes artifact-value economics

ATP's adaptive RaptorQ machinery frames repair as ROI, not a universal default.

FrankenSymPy repair decisions should use:

- recomputation cost;
- expected time to regenerate;
- verifier cost;
- transfer and storage cost;
- artifact retention horizon;
- observed failure/loss regime;
- number and independence of replicas;
- resume criticality;
- privacy class;
- current CPU, memory, and network pressure.

A checkpoint from a multi-hour Gröbner run may merit repair symbols. A tiny canonical term does not.

Telemetry fields that are estimates must remain labeled `Estimated`; they cannot silently enter a verified durability claim.

## 17. Lab/live dual-run becomes symbolic execution differential testing

Asupersync distinguishes scenario-family identity from execution-instance identity and carries canonical seed lineage, lab/live adapters, replay policy, trace fingerprints, schedule hashes, and repro commands.

FrankenSymPy should use the same two-level identity:

```text
ScenarioFamily
├── semantic surface and comparator version
├── generated problem grammar
└── adversarial invariant

ExecutionInstance
├── exact seed and substream plan
├── runtime mode and architecture
├── profile/context/registry IDs
├── schedule/trace fingerprint
└── artifact/repro pointer
```

Applications:

- deterministic runtime versus production runtime;
- scalar versus optimized verifier;
- local versus remote generator;
- persistent versus ephemeral execution;
- strict versus latency-adaptive planning;
- upstream SymPy versus compatibility shell.

Shrinking preserves scenario-family identity while producing a new execution instance and minimized artifact.

## 18. Conformal and e-process machinery remains operational

Asupersync's conformal code explicitly relies on exchangeability for finite-sample coverage. Its e-processes state a null model and use anytime-valid thresholds.

FrankenSymPy should inherit the discipline, not merely the terminology.

Every monitor records:

- population and filtration;
- calibration set and update policy;
- null and alternative;
- conformity score or betting strategy;
- subgroup selection;
- reset and quarantine policy;
- action thresholds;
- missing/refused/timeout inclusion;
- evidence and policy version.

Appropriate actions:

- freeze selector learning;
- revert to a safe baseline;
- increase shadow verification;
- quarantine a worker or strategy;
- widen a repair safety margin;
- open a discrepancy.

Forbidden action:

- accepting or rejecting an individual mathematical claim.

## 19. Hedging applies to generators, not truth

Asupersync's hedge combinator reduces tail latency but explicitly notes that standalone loser dropping differs from runtime-enforced draining.

FrankenSymPy may hedge:

- remote candidate generation;
- modular prime batches;
- expensive heuristic integration search;
- proof compression;
- cache or artifact retrieval.

It must not make “first completion” the strict semantic arbitration rule. A hedge winner is still a candidate. The verifier and deterministic result-form policy decide publication. Losers are drained under the owning region.

## 20. New derived concepts

The combined audit suggests several mechanisms not explicit in the original plan.

### 20.1 Verifier-complete cut

Compute the minimum closed set of artifact objects needed to verify a claim. This cut is:

- the default portable capsule;
- the highest transfer priority;
- the unit of cache admission;
- the unit of repair-value analysis;
- the root for garbage collection;
- the boundary for external checker compatibility.

### 20.2 Proof availability frontier

Track which subclaims have complete local verification closure as artifacts stream or compute. A UI or agent can consume verified partial structure without mistaking incomplete global coverage for completion.

### 20.3 Evidence escrow

Candidate bytes, certificate bytes, and publication rights remain in a private escrow namespace. Only the verifier can release the exact claim/value/certificate triple into a verified cache or branch.

### 20.4 Merkle-delta derivation exchange

Agent branches and distributed workers synchronize derivation graphs by content identity and changed Merkle subtrees. Textual diffs become a human view, not the transfer primitive.

### 20.5 Resume calculus

Each long algorithm declares a frontier decomposition into:

- verified durable subclaims;
- unverified candidate state;
- deterministic remaining frontier;
- leased random/subgoal ranges;
- resource accounting.

Only the first and the canonical frontier are resumable authority.

### 20.6 Proof-unlocking utility

Remote and transport scheduling can estimate which missing artifact yields the largest increase in independently verifiable claim coverage per byte or unit compute.

This is more useful than generic “chunk rarity” for mathematical workloads.

## 21. Adopt, adapt, reject summary

### Adopt directly

- explicit capability narrowing;
- region ownership and finalizers;
- lab/live scenario identity and replay metadata;
- canonical manifests and critical-field handling;
- content-addressed object graphs;
- append-only staged journals;
- fail-closed bounded verification;
- preflight plans and deviation reports;
- deterministic resource governance;
- delta CAS and Merkle diff concepts.

### Adapt behind FrankenSymPy contracts

- ATP object kinds into mathematical artifact kinds;
- transfer proof bundles into verifier-complete capsules;
- transfer priorities into proof-availability priorities;
- multi-source repair scheduling into untrusted subclaim scheduling;
- adaptive RaptorQ into artifact-value repair policy;
- hedging into generator-only speculation;
- conformal/e-process code into registry-governed operational monitors;
- Slepian–Wolf into an optional artifact-delta research lane.

### Reject

- any runtime dependency in portable verifier crates;
- using transport identity as mathematical identity;
- using signatures, worker trust, repair, or replay as proof;
- first-completion semantic arbitration;
- hidden ambient time, entropy, I/O, or network effects;
- resuming from merely computed or merely written state;
- importing ATP's entire dependency surface into the symbolic core;
- claiming universal cancellation through arbitrary Python or foreign code.

## 22. FrankenSymPy workstream consequences

The architecture should add or strengthen:

1. an embeddable verifier workstream and public compatibility claim;
2. a canonical mathematical artifact protocol;
3. an asupersync ATP adapter that is optional and higher-layer;
4. staged symbolic journals and evidence escrow;
5. proof-availability scheduling;
6. semantic object and Merkle-delta synchronization;
7. local minimal-consumer verifier gates;
8. lab/live dual-run and schedule exploration for portfolios and publication;
9. artifact-value repair experiments;
10. an optional Slepian–Wolf research gate.

The portable verifier and artifact core must precede network transport. A mathematically complete capsule should verify equally from memory, disk, browser storage, ATP, or another transport.
